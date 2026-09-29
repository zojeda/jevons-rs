//! The inspector and settings window, on dioxus-native (Blitz): the live context and the route it
//! takes through the flow tree, the recent takes, the flow tree, the settings and the models.
//! Closing it hides it; the tray reopens it.
//!
//! The window lives on the main thread's winit loop, with the feedback bubble while a take runs.
//! The agent and background work call [`wake`], which re-renders from the shared
//! [`View`](crate::agent::View).

mod app;
mod bubble;
mod components;
mod context;
mod flows;
mod models;
mod settings;
mod takes;

use crate::agent::{Command, SharedView};
use anyrender_vello::{VelloRendererOptions, VelloWindowRenderer};
use blitz_shell::{BlitzApplication, BlitzShellEvent, View, WindowConfig};
use dioxus::prelude::*;
use dioxus_native::{DioxusDocument, DocumentConfig};
use std::sync::{Mutex, OnceLock};
use tokio::sync::mpsc::UnboundedSender;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::window::{Window, WindowId, WindowLevel};

/// Asks the window to re-render from the shared view.
#[derive(Debug)]
struct Refresh;

static PROXY: OnceLock<Mutex<EventLoopProxy<BlitzShellEvent>>> = OnceLock::new();

/// Re-renders the window (and shows or closes it when the view asks); callable from any thread.
pub fn wake() {
    if let Some(proxy) = PROXY.get() {
        let _ = proxy
            .lock()
            .expect("the window proxy lock")
            .send_event(BlitzShellEvent::embedder_event(Refresh));
    }
}

/// What the window's components share.
#[derive(Clone)]
pub struct Ctx {
    pub view: SharedView,
    pub commands: UnboundedSender<Command>,
}

impl Ctx {
    pub fn send(&self, command: Command) {
        let _ = self.commands.send(command);
    }
}

/// Runs the window on this (the main) thread until Quit. `start` runs once the event loop exists,
/// so the agent it starts can already [`wake`] the window.
pub fn run(
    view: SharedView,
    commands: UnboundedSender<Command>,
    start: impl FnOnce(),
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = blitz_shell::create_default_event_loop::<BlitzShellEvent>();
    let proxy = event_loop.create_proxy();
    let _ = PROXY.set(Mutex::new(proxy.clone()));
    start();

    let visible = !view.lock().expect("the view lock").tray_running;
    let ctx = Ctx {
        view: view.clone(),
        commands,
    };
    let mut vdom = VirtualDom::new(app::App);
    vdom.insert_any_root_context(Box::new(ctx.clone()));
    let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
    doc.add_user_agent_stylesheet(include_str!("style.css"));
    doc.initial_build();
    let renderer = VelloWindowRenderer::with_options(VelloRendererOptions {
        base_color: peniko::Color::from_rgb8(10, 10, 10),
        ..Default::default()
    });
    let icon =
        winit::window::Icon::from_rgba(jevons_desktop_core::icons::app_icon(64), 64, 64).ok();
    let attributes = Window::default_attributes()
        .with_title("jevons")
        .with_inner_size(LogicalSize::new(860.0, 700.0))
        .with_min_inner_size(LogicalSize::new(560.0, 420.0))
        .with_visible(visible)
        .with_window_icon(icon);
    let mut inner = BlitzApplication::new(proxy);
    inner.add_window(WindowConfig::with_attributes(
        Box::new(doc),
        renderer,
        attributes,
    ));
    view.lock().expect("the view lock").window_visible = visible;
    let mut shell = Shell {
        inner,
        view,
        ctx,
        main: None,
        bubble: None,
    };
    event_loop.run_app(&mut shell)?;
    Ok(())
}

struct Shell {
    inner: BlitzApplication<VelloWindowRenderer>,
    view: SharedView,
    ctx: Ctx,
    /// The inspector and settings window.
    main: Option<WindowId>,
    /// The feedback bubble, while it shows.
    bubble: Option<WindowId>,
}

/// Where the bubble goes: just above the tray icon (or below it, for a taskbar at the top), and
/// at the bottom right of the screen when the icon's place is unknown.
fn place_bubble(
    event_loop: &ActiveEventLoop,
) -> Option<(PhysicalPosition<i32>, f64, bubble::Anchor)> {
    let icon = crate::tray::icon_rect();
    let monitor = event_loop
        .available_monitors()
        .find(|m| {
            icon.is_some_and(|(x, y, _, _)| {
                let (p, size) = (m.position(), m.size());
                x >= f64::from(p.x)
                    && x < f64::from(p.x) + f64::from(size.width)
                    && y >= f64::from(p.y)
                    && y < f64::from(p.y) + f64::from(size.height)
            })
        })
        .or_else(|| event_loop.primary_monitor())
        .or_else(|| event_loop.available_monitors().next())?;
    let scale = monitor.scale_factor();
    let (origin, size) = (monitor.position(), monitor.size());
    let (left, top) = (f64::from(origin.x), f64::from(origin.y));
    let (width, height) = (f64::from(size.width), f64::from(size.height));
    let (w, h) = (bubble::SIZE.0 * scale, bubble::SIZE.1 * scale);
    let gap = 6.0 * scale;
    let (center, y, icon_below) = match icon {
        Some((x, y, iw, ih)) => {
            let below = y - top > height / 2.0;
            let center = x + f64::from(iw) / 2.0;
            (
                center,
                if below {
                    y - h - gap
                } else {
                    y + f64::from(ih) + gap
                },
                below,
            )
        }
        None => (
            left + width - w / 2.0 - 16.0 * scale,
            top + height - h - 56.0 * scale,
            true,
        ),
    };
    let x = (center - w / 2.0).clamp(left + 8.0 * scale, left + width - w - 8.0 * scale);
    let anchor = bubble::Anchor {
        tail_x: (center - x) / scale,
        icon_below,
    };
    Some((PhysicalPosition::new(x as i32, y as i32), scale, anchor))
}

impl Shell {
    /// Opens the feedback bubble without taking the focus from the application being dictated
    /// into. It is a new window each time: winit shows a window without activating it only once.
    fn open_bubble(&mut self, event_loop: &ActiveEventLoop) {
        let Some((position, scale, anchor)) = place_bubble(event_loop) else {
            return;
        };
        let mut vdom = VirtualDom::new(bubble::Bubble);
        vdom.insert_any_root_context(Box::new(self.ctx.clone()));
        vdom.insert_any_root_context(Box::new(anchor));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("bubble.css"));
        doc.initial_build();
        let renderer = VelloWindowRenderer::with_options(VelloRendererOptions {
            base_color: peniko::Color::from_rgb8(14, 14, 14),
            ..Default::default()
        });
        let size = PhysicalSize::new(
            (bubble::SIZE.0 * scale).round() as u32,
            (bubble::SIZE.1 * scale).round() as u32,
        );
        #[allow(unused_mut)]
        let mut attributes = Window::default_attributes()
            .with_title("jevons feedback")
            .with_decorations(false)
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_active(false)
            .with_inner_size(size)
            .with_position(position);
        #[cfg(windows)]
        {
            use winit::platform::windows::WindowAttributesExtWindows;
            attributes = attributes.with_skip_taskbar(true);
        }
        let mut view = View::init(
            WindowConfig::with_attributes(Box::new(doc), renderer, attributes),
            event_loop,
            &self.inner.proxy,
        );
        view.resume();
        // Clicks go to the application underneath.
        let _ = view.window.set_cursor_hittest(false);
        let id = view.window_id();
        self.inner.windows.insert(id, view);
        self.bubble = Some(id);
    }

    fn refresh(&mut self, event_loop: &ActiveEventLoop) {
        let (quit, show, bubble) = {
            let mut view = self.view.lock().expect("the view lock");
            let bubble = view.feedback.is_some() && view.config.dictation.live_feedback;
            (view.quit, std::mem::take(&mut view.show_window), bubble)
        };
        if quit {
            event_loop.exit();
            return;
        }
        match (bubble, self.bubble) {
            (true, None) => self.open_bubble(event_loop),
            (false, Some(id)) => {
                self.bubble = None;
                self.inner.windows.remove(&id);
            }
            _ => {}
        }
        for (id, window) in self.inner.windows.iter_mut() {
            if show && Some(*id) == self.main {
                window.window.set_visible(true);
                window.window.set_minimized(false);
                window.window.focus_window();
                self.view.lock().expect("the view lock").window_visible = true;
            }
            let doc = window.downcast_doc_mut::<DioxusDocument>();
            doc.vdom.mark_dirty(ScopeId::APP);
            window.poll();
            window.request_redraw();
        }
    }
}

impl ApplicationHandler<BlitzShellEvent> for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.resumed(event_loop);
        if self.main.is_none() {
            self.main = self.inner.windows.keys().next().copied();
        }
        self.refresh(event_loop);
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.suspended(event_loop);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) && Some(id) == self.bubble {
            return;
        }
        if matches!(event, WindowEvent::CloseRequested) {
            // Hide instead of closing: the tray keeps running and reopens the window.
            if let Some(window) = self.inner.windows.get(&id) {
                window.window.set_visible(false);
            }
            self.view.lock().expect("the view lock").window_visible = false;
            return;
        }
        self.inner.window_event(event_loop, id, event);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: BlitzShellEvent) {
        if let BlitzShellEvent::Embedder(payload) = &event
            && payload.is::<Refresh>()
        {
            self.refresh(event_loop);
            return;
        }
        self.inner.user_event(event_loop, event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::View;
    use blitz_dom::Document as _;
    use jevons_desktop_core::config::DesktopConfig;
    use jevons_desktop_core::context::{AppInfo, ContextSnapshot, Element as Focused, WindowInfo};
    use jevons_desktop_core::flow::spec::Output;
    use jevons_desktop_core::flow::walk::{self, Leaf};
    use jevons_desktop_core::flow::{Catalog, FlowTree, defaults};
    use jevons_desktop_core::pipeline::{Trace, TranscriptionPath};
    use jevons_desktop_core::platform::{Action, DeliveryMethod, DeliveryOutcome};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// Which page the test root shows; switching it re-renders like changing tabs.
    static PAGE: AtomicU8 = AtomicU8::new(0);

    fn root() -> Element {
        let frozen = use_signal(|| false);
        match PAGE.load(Ordering::Relaxed) {
            0 => rsx! { context::ContextPage { rev: 0, frozen } },
            1 => rsx! { models::ModelsPage { rev: 1 } },
            2 => rsx! { settings::SettingsPage { rev: 2 } },
            3 => rsx! { takes::TakesPage { rev: 3 } },
            4 => rsx! { flows::FlowsPage { rev: 4 } },
            _ => rsx! { app::App {} },
        }
    }

    /// A models folder like the one on a machine mid-download: Parakeet ready, Gemma partial.
    fn models_folder() -> std::path::PathBuf {
        let folder = std::env::temp_dir().join(format!("jevons-ui-{}", std::process::id()));
        let parakeet = folder.join("parakeet-tdt-0.6b-v3");
        std::fs::create_dir_all(&parakeet).unwrap();
        std::fs::write(
            parakeet.join(jevons_desktop_core::download::COMPLETE_MARKER),
            b"",
        )
        .unwrap();
        let gemma = folder.join("diffusiongemma-26b-a4b-q4_k_m");
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::write(gemma.join("model.gguf.part"), b"partial").unwrap();
        folder
    }

    fn flows() -> Arc<FlowTree> {
        Arc::new(FlowTree::load(&defaults::builtin(), &Catalog::default()))
    }

    /// A finished take numbered `take`, like the agent records.
    fn trace(take: u64, context: &ContextSnapshot, flows: &FlowTree) -> Trace {
        Trace {
            take,
            turn: None,
            started_at_ms: take,
            context: context.clone(),
            audio_seconds: 1.5,
            transcription: Some(TranscriptionPath::Realtime),
            transcript: "hello world".into(),
            entry: "/".into(),
            flow: walk::preview(flows, context, flows.root()),
            leaf: Some(Leaf {
                node: "_actions/insert".into(),
                text: "Hello world.".into(),
                output: Output::Target,
                action: Action::Insert,
                delivery: DeliveryMethod::Paste,
            }),
            generation: None,
            output: "Hello world.".into(),
            delivery: Some(DeliveryOutcome::Delivered {
                method: DeliveryMethod::Paste,
            }),
            timings: vec![("transcribe".into(), 800)],
            notes: vec!["a note".into()],
            error: None,
        }
    }

    /// Re-render counter for [`takes_root`], as the window's App passes a new revision each time.
    static REV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn takes_root() -> Element {
        let rev = REV.fetch_add(1, Ordering::Relaxed);
        rsx! { takes::TakesPage { rev } }
    }

    #[test]
    fn new_takes_arriving_while_the_takes_page_shows_rebuild_in_blitz() {
        let folder = std::env::temp_dir().join(format!("jevons-ui-takes-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(takes_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.initial_build();
        let flows = flows();
        // New takes go on top, as the agent adds them, and the oldest fall off after 50.
        for take in 2..80 {
            {
                let mut view = view.lock().unwrap();
                let context = view.context.clone().unwrap();
                view.traces.push_front(trace(take, &context, &flows));
                view.traces.truncate(50);
            }
            doc.vdom.mark_dirty(ScopeId::APP);
            doc.poll(None);
        }
    }

    #[test]
    fn the_feedback_bubble_follows_a_live_take_to_its_end_in_blitz() {
        use crate::agent::Feedback;
        let folder = std::env::temp_dir().join(format!("jevons-ui-bubble-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        // The bubble is its window's root, as the shell opens it.
        let mut vdom = VirtualDom::new(bubble::Bubble);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        vdom.insert_any_root_context(Box::new(bubble::Anchor {
            tail_x: 300.0,
            icon_below: true,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("bubble.css"));
        doc.initial_build();
        let feedback = Feedback {
            take: 1,
            live: true,
            status: "Listening…".into(),
            ..Feedback::default()
        };
        type Step = Box<dyn Fn(&mut Feedback)>;
        let steps: Vec<Step> = vec![
            Box::new(|f| f.partial.push_str("hello")),
            Box::new(|f| f.partial.push_str(" there")),
            Box::new(|f| {
                f.heard = "Hello there.".into();
                f.partial.clear();
            }),
            Box::new(|f| f.steps.push("Route dictate".into())),
            Box::new(|f| f.output = "Hello there!".into()),
            Box::new(|f| {
                f.done = true;
                f.status = "Inserted".into();
            }),
        ];
        view.lock().unwrap().feedback = Some(feedback);
        for step in steps {
            step(view.lock().unwrap().feedback.as_mut().unwrap());
            doc.vdom.mark_dirty(ScopeId::APP);
            doc.poll(None);
        }
        let text = doc.root_element().text_content();
        assert!(text.contains("Hello there."), "{text}");
        assert!(text.contains("Route dictate"), "{text}");
        assert!(text.contains("Inserted"), "{text}");
        // The take ends and the bubble empties.
        view.lock().unwrap().feedback = None;
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
    }

    fn context_root() -> Element {
        let rev = REV.fetch_add(1, Ordering::Relaxed);
        let frozen = use_signal(|| false);
        rsx! { context::ContextPage { rev, frozen } }
    }

    #[test]
    fn routes_changing_as_the_focused_app_changes_rebuild_in_blitz() {
        let folder = std::env::temp_dir().join(format!("jevons-ui-order-{}", std::process::id()));
        let flows = flows();
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(context_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.initial_build();
        // Each app takes another branch of dictate, so the route and its checks change.
        for app in [
            "outlook.exe",
            "slack.exe",
            "code.exe",
            "notepad.exe",
            "slack.exe",
            "outlook.exe",
        ]
        .repeat(3)
        {
            {
                let mut view = view.lock().unwrap();
                let mut context = view.context.clone().unwrap();
                context.app.process_name = app.into();
                let dictate = flows.find("dictate").unwrap();
                view.route = walk::preview(&flows, &context, dictate);
                view.context = Some(context);
            }
            doc.vdom.mark_dirty(ScopeId::APP);
            doc.poll(None);
        }
    }

    /// A view with a context, its route and a finished take, so every section renders.
    fn view(folder: &std::path::Path) -> View {
        let mut config = DesktopConfig::default();
        config.models.folder = Some(folder.to_path_buf());
        let context = ContextSnapshot {
            app: AppInfo {
                process_name: "notepad.exe".into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: "notes.txt - Notepad".into(),
                handle: Some(1),
                ..WindowInfo::default()
            },
            focused: Some(Focused {
                role: "Document".into(),
                selection: Some("hello".into()),
                ..Focused::default()
            }),
            ..ContextSnapshot::default()
        };
        let flows = flows();
        let route = walk::preview(&flows, &context, flows.root());
        View {
            config,
            traces: [trace(1, &context, &flows)].into(),
            context: Some(context),
            route,
            flows,
            ..View::default()
        }
    }

    #[test]
    fn switching_between_every_page_rebuilds_in_blitz() {
        let folder = models_folder();
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(root);
        vdom.insert_any_root_context(Box::new(Ctx { view, commands }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.initial_build();
        // Every page after every other, and back: pages are removed and rebuilt as with tabs.
        let mut sequence = Vec::new();
        for a in 0..=5 {
            for b in 0..=5 {
                sequence.extend([a, b, a]);
            }
        }
        for page in sequence {
            PAGE.store(page, Ordering::Relaxed);
            doc.vdom.mark_dirty(ScopeId::APP);
            doc.poll(None);
        }
        std::fs::remove_dir_all(folder).unwrap();
    }
}
