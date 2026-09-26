//! The platform layers for this OS. Windows reads the focused element through UI Automation and
//! types with SendInput; elsewhere the active window is known but the focused element is not
//! yet (AT-SPI and the macOS Accessibility API come later), and text stays on the clipboard.

use jevons_desktop_core::context::{AppInfo, ContextSnapshot, Privacy, WindowInfo};
use jevons_desktop_core::platform::{
    ContextProvider, DeliveryOutcome, DeliveryRequest, PlatformError, SinkCapabilities, TextSink,
};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(windows)]
mod windows;

/// The context provider for this platform.
pub fn context_provider() -> Box<dyn ContextProvider> {
    #[cfg(windows)]
    return Box::new(windows::UiaContext);
    #[cfg(not(windows))]
    Box::new(WindowContext)
}

/// The text sink for this platform.
pub fn text_sink() -> Box<dyn TextSink> {
    #[cfg(windows)]
    return Box::new(windows::WindowsSink::new());
    #[cfg(not(windows))]
    Box::new(ClipboardSink::default())
}

/// The active window, with a handle stable for this session.
pub(crate) fn active_window() -> Option<(active_win_pos_rs::ActiveWindow, u64)> {
    let window = active_win_pos_rs::get_active_window().ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (window.process_id, &window.window_id).hash(&mut hasher);
    let handle = hasher.finish().max(1);
    Some((window, handle))
}

/// The application and window part of a snapshot.
pub(crate) fn window_snapshot() -> Result<ContextSnapshot, PlatformError> {
    let (window, handle) = active_window()
        .ok_or_else(|| PlatformError::Failed("Cannot read the active window".into()))?;
    let process_name = window
        .process_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| window.app_name.clone());
    Ok(ContextSnapshot {
        captured_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64),
        app: AppInfo {
            process_name,
            exe: Some(window.process_path.display().to_string()),
            pid: u32::try_from(window.process_id).ok(),
        },
        window: WindowInfo {
            title: window.title,
            class: None,
            handle: Some(handle),
        },
        ..ContextSnapshot::default()
    })
}

/// The active application and window only.
#[cfg_attr(windows, allow(dead_code))]
pub struct WindowContext;

impl ContextProvider for WindowContext {
    fn name(&self) -> &'static str {
        "active window"
    }

    fn snapshot(&self, privacy: &Privacy) -> Result<ContextSnapshot, PlatformError> {
        let mut snapshot = window_snapshot()?;
        snapshot
            .errors
            .push("The focused element is not readable on this platform yet".into());
        Ok(snapshot.sanitized(privacy))
    }
}

/// Copies the text; typing into other applications is not available on this platform yet.
#[derive(Default)]
#[cfg_attr(windows, allow(dead_code))]
pub struct ClipboardSink {
    clipboard: Option<arboard::Clipboard>,
}

impl TextSink for ClipboardSink {
    fn name(&self) -> &'static str {
        "clipboard"
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities::default()
    }

    fn foreground_window(&self) -> Option<u64> {
        active_window().map(|(_, handle)| handle)
    }

    fn keys_down(&self) -> bool {
        false
    }

    fn deliver(&mut self, request: &DeliveryRequest) -> Result<DeliveryOutcome, PlatformError> {
        self.copy(&request.text)?;
        Ok(DeliveryOutcome::OnClipboard {
            reason: "typing into other applications is not available on this platform yet".into(),
        })
    }

    fn copy(&mut self, text: &str) -> Result<(), PlatformError> {
        let clipboard = match &mut self.clipboard {
            Some(clipboard) => clipboard,
            None => self.clipboard.insert(
                arboard::Clipboard::new().map_err(|e| PlatformError::Failed(e.to_string()))?,
            ),
        };
        clipboard
            .set_text(text)
            .map_err(|e| PlatformError::Failed(e.to_string()))
    }
}
