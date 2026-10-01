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
mod interface;
mod markdown;
mod models;
mod settings;
mod takes;
mod workbench;

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
        bubble_size: bubble::SIZE,
        bubble_clicks: false,
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
    /// Its size in logical pixels: larger while it shows an answer.
    bubble_size: (f64, f64),
    /// Whether the bubble takes clicks and the wheel (an answer or a confirmation).
    bubble_clicks: bool,
}

/// Where a bubble of `size` goes: just above the tray icon (or below it, for a taskbar at the
/// top), and at the bottom right of the screen when the icon's place is unknown.
fn place_bubble(
    event_loop: &ActiveEventLoop,
    bubble_size: (f64, f64),
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
    let (w, h) = (bubble_size.0 * scale, bubble_size.1 * scale);
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
    // A tall bubble never leaves the screen.
    let y = y.clamp(top + 8.0 * scale, (top + height - h - 8.0 * scale).max(top));
    let anchor = bubble::Anchor {
        tail_x: (center - x) / scale,
        icon_below,
        width: bubble_size.0,
    };
    Some((PhysicalPosition::new(x as i32, y as i32), scale, anchor))
}

impl Shell {
    /// Opens the feedback bubble without taking the focus from the application being dictated
    /// into. It is a new window each time: winit shows a window without activating it only once.
    fn open_bubble(&mut self, event_loop: &ActiveEventLoop, size: (f64, f64)) {
        let Some((position, scale, anchor)) = place_bubble(event_loop, size) else {
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
        let physical = PhysicalSize::new(
            (size.0 * scale).round() as u32,
            (size.1 * scale).round() as u32,
        );
        #[allow(unused_mut)]
        let mut attributes = Window::default_attributes()
            .with_title("jevons feedback")
            .with_decorations(false)
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_active(false)
            .with_inner_size(physical)
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
        self.bubble_size = size;
    }

    fn refresh(&mut self, event_loop: &ActiveEventLoop) {
        let (quit, show, bubble, clicks) = {
            let mut view = self.view.lock().expect("the view lock");
            // A call waiting for confirmation shows even with live feedback off.
            let asking = view.feedback.as_ref().is_some_and(|f| f.confirm.is_some());
            let message = view.feedback.as_ref().is_some_and(|f| f.message);
            // An answer is for reading: it shows even with live feedback off.
            let answer = view.feedback.as_ref().is_some_and(|f| f.answer);
            let bubble = view.feedback.is_some()
                && (view.config.dictation.live_feedback || asking || message || answer);
            // An answer takes the wheel as soon as it streams in, and clicks once it is done.
            let clicks = view
                .feedback
                .as_ref()
                .is_some_and(|f| f.confirm.is_some() || f.answer);
            let size = if answer && !asking {
                bubble::ANSWER_SIZE
            } else {
                bubble::SIZE
            };
            (
                view.quit,
                std::mem::take(&mut view.show_window),
                bubble.then_some(size),
                clicks,
            )
        };
        if quit {
            event_loop.exit();
            return;
        }
        match (bubble, self.bubble) {
            (Some(size), None) => {
                self.open_bubble(event_loop, size);
                self.bubble_clicks = false;
            }
            // A new size is a new window: winit shows a window without activating it only once.
            (Some(size), Some(id)) if size != self.bubble_size => {
                self.inner.windows.remove(&id);
                self.open_bubble(event_loop, size);
                self.bubble_clicks = false;
            }
            (None, Some(id)) => {
                self.bubble = None;
                self.inner.windows.remove(&id);
            }
            _ => {}
        }
        if let Some(window) = self.bubble.and_then(|id| self.inner.windows.get(&id))
            && clicks != self.bubble_clicks
        {
            let _ = window.window.set_cursor_hittest(clicks);
            self.bubble_clicks = clicks;
        }
        // An element the interface browser revealed, to scroll into view once rendered.
        let reveal = self
            .view
            .lock()
            .expect("the view lock")
            .interface
            .as_mut()
            .and_then(|b| b.reveal.take());
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
            if Some(*id) == self.main
                && let Some(row) = &reveal
            {
                interface::scroll_into_view(window.downcast_doc_mut::<DioxusDocument>(), row);
            }
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
            calls: Vec::new(),
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
            width: bubble::SIZE.0,
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
            Box::new(|f| {
                f.working = true;
                f.stages.push(crate::agent::StageView {
                    kind: jevons_desktop_core::pipeline::StageKind::Deciding,
                    label: "what to do".into(),
                    choices: vec!["ask".into(), "dictate".into()],
                    detail: String::new(),
                    chosen: None,
                    ok: None,
                });
            }),
            Box::new(|f| f.frame += 5),
            Box::new(|f| {
                let stage = f.stages.last_mut().unwrap();
                stage.chosen = Some("dictate".into());
                stage.detail = "0.92".into();
                stage.ok = Some(true);
                f.stages.push(crate::agent::StageView {
                    kind: jevons_desktop_core::pipeline::StageKind::Writing,
                    label: "text".into(),
                    choices: Vec::new(),
                    detail: "writing".into(),
                    chosen: None,
                    ok: None,
                });
            }),
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
        assert!(
            text.contains("what to do") && text.contains("dictate") && text.contains("0.92"),
            "{text}"
        );
        assert!(text.contains("Inserted"), "{text}");
        // A tool call waits, then the take answers in the bubble.
        view.lock().unwrap().feedback.as_mut().unwrap().confirm = Some(crate::agent::PendingCall {
            tool: "notes:create_note".into(),
            arguments: "{\"title\": \"Launch\"}".into(),
        });
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        let text = doc.root_element().text_content();
        assert!(
            text.contains("Run notes:create_note?") && text.contains("Cancel (Esc)"),
            "{text}"
        );
        {
            let mut view = view.lock().unwrap();
            let feedback = view.feedback.as_mut().unwrap();
            feedback.confirm = None;
            feedback.answer = true;
            feedback.output = "## Launch\n\nThe launch is **on Friday**:\n\n1. build\n2. ship\n\n\
                               | who | when |\n|---|---|\n| Ana | `Fri` |\n"
                .into();
            feedback.window = Some(7);
        }
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        let text = doc.root_element().text_content();
        assert!(
            text.contains("The launch is on Friday:")
                && text.contains("1.build")
                && text.contains("Ana")
                && !text.contains("**")
                && text.contains("Insert")
                && text.contains("Copy"),
            "{text}"
        );
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
    /// The workbench with the root's `slack_messages` chosen, as the "Edit" of a reading picks it.
    fn workbench_root() -> Element {
        let rev = REV.fetch_add(1, Ordering::Relaxed);
        let chosen = use_signal(|| Some(workbench::key("", "slack_messages")));
        rsx! { workbench::Workbench { rev, chosen } }
    }

    #[test]
    fn the_extract_workbench_loads_the_chosen_extract_and_shows_a_trial_in_blitz() {
        use jevons_desktop_core::flow::extract::{Extracted, Trial};
        let folder = std::env::temp_dir().join(format!("jevons-ui-bench-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, mut received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(workbench_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.initial_build();
        doc.poll(None);
        let html = doc.root_element().outer_html();
        assert!(
            html.contains("message-list_"),
            "the expression loads: {html}"
        );
        assert!(html.contains("Save to decide.toml"), "{html}");
        view.lock().unwrap().trial = Some(crate::agent::TrialView {
            name: "slack_messages".into(),
            xpath: "//ListItem".into(),
            window: "slack.exe · general".into(),
            trial: Trial {
                found: Some(Extracted {
                    value: serde_json::json!(["Ana: hi", "Bo: hello"]),
                    matches: 2,
                    note: None,
                }),
                matched: vec!["ListItem \"Ana: hi\"".into()],
                ms: 12,
                ..Trial::default()
            },
        });
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        let text = doc.root_element().text_content();
        assert!(
            text.contains("2 matches") && text.contains("Bo: hello"),
            "{text}"
        );
        assert!(
            text.contains("(for an earlier version of the expression)"),
            "{text}"
        );
        assert!(received.try_recv().is_err(), "nothing is tried until asked");
    }

    /// The interface browser beside the workbench, sharing the draft as the Context page does;
    /// the draft starts with an expression, as **Try in workbench** sets it.
    fn interface_root() -> Element {
        let rev = REV.fetch_add(1, Ordering::Relaxed);
        let chosen = use_signal(|| None::<String>);
        let draft = use_signal(|| Some("//Edit[has-class(@class, 'ql-editor')]".to_string()));
        rsx! {
            workbench::Workbench { rev, chosen, draft }
            interface::Interface { rev, draft }
        }
    }

    #[test]
    fn the_interface_browser_shows_the_opened_tree_and_hands_a_selector_to_the_workbench() {
        use crate::agent::{Command, InterfaceView};
        use jevons_desktop_core::interface as browse;
        use jevons_desktop_core::platform::ContextInspector;
        use jevons_desktop_core::recorded::RecordedInspector;
        let folder = std::env::temp_dir().join(format!("jevons-ui-iface-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, mut received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(interface_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.initial_build();
        doc.poll(None);
        // The draft becomes a new expression, tried at once; the browser reads its window.
        let mut sent = Vec::new();
        while let Ok(command) = received.try_recv() {
            sent.push(command);
        }
        assert!(
            sent.iter().any(|c| matches!(c, Command::TryExtract(r)
                if r.file.is_none() && r.spec.xpath.contains("ql-editor"))),
            "{sent:?}"
        );
        assert!(sent.iter().any(|c| matches!(c, Command::InterfaceLoad)));
        // The agent's reads, from the recorded Slack window.
        let inspector = RecordedInspector::new(
            serde_json::from_str(include_str!(
                "../../../../examples/desktop/trees/slack.json"
            ))
            .unwrap(),
        );
        let window = inspector.windows().unwrap().remove(0);
        let opened =
            browse::open_below(&inspector, &window.id, 64, 10_000, &Default::default()).unwrap();
        let mut browser = InterfaceView {
            generation: 1,
            window: window.clone(),
            key: "slack.exe · general (Channel) - Acme - Slack".into(),
            ..InterfaceView::default()
        };
        for (parent, level) in opened.levels {
            browser.open.insert(parent.clone());
            browser.levels.insert(parent, level);
        }
        let composer = browser
            .levels
            .values()
            .flat_map(|l| l.elements.iter())
            .find(|e| e.role == "Edit")
            .cloned()
            .unwrap();
        browser.selectors = Some(browse::selectors(&inspector, &window, &composer.id));
        browser.selected = Some(composer);
        view.lock().unwrap().interface = Some(browser);
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        let text = doc.root_element().text_content();
        assert!(
            text.contains("Message #general") && text.contains("ql-editor"),
            "{text}"
        );
        assert!(text.contains("Try in workbench"), "{text}");
        // The tab shows another window (Notepad) than the tree's.
        assert!(text.contains("This tree is of slack.exe"), "{text}");
    }

    /// The interface browser alone, as the Context page shows it.
    fn browser_root() -> Element {
        let rev = REV.fetch_add(1, Ordering::Relaxed);
        let draft = use_signal(|| None::<String>);
        rsx! { interface::Interface { rev, draft } }
    }

    /// Clicks the middle of the element `selector` finds, as the window would: the pointer moves
    /// there, then the button goes down and up.
    fn click(doc: &mut DioxusDocument, selector: &str) {
        use blitz_traits::events::{
            BlitzMouseButtonEvent, MouseEventButton, MouseEventButtons, UiEvent,
        };
        doc.resolve(0.0);
        let node = doc
            .query_selector(selector)
            .unwrap()
            .unwrap_or_else(|| panic!("nothing matches {selector}"));
        let (x, y) = {
            let node = doc.get_node(node).unwrap();
            let at = node.absolute_position(0.0, 0.0);
            let size = node.final_layout.size;
            assert!(
                size.width >= 16.0 && size.height >= 16.0,
                "{selector}: {size:?}"
            );
            (at.x + size.width / 2.0, at.y + size.height / 2.0)
        };
        let event = || BlitzMouseButtonEvent {
            x,
            y,
            button: MouseEventButton::Main,
            buttons: MouseEventButtons::Primary,
            mods: Default::default(),
        };
        doc.handle_ui_event(UiEvent::MouseMove(event()));
        doc.handle_ui_event(UiEvent::MouseDown(event()));
        doc.handle_ui_event(UiEvent::MouseUp(event()));
        doc.poll(None);
    }

    #[test]
    fn interface_rows_toggle_by_their_chevron_and_select_by_their_label_with_real_clicks() {
        use crate::agent::{Command, InterfaceSearch, InterfaceView};
        use blitz_traits::shell::{ColorScheme, Viewport};
        use jevons_desktop_core::interface as browse;
        use jevons_desktop_core::platform::ContextInspector;
        use jevons_desktop_core::recorded::RecordedInspector;
        let inspector = RecordedInspector::new(
            serde_json::from_str(include_str!(
                "../../../../examples/desktop/trees/slack.json"
            ))
            .unwrap(),
        );
        let window = inspector.windows().unwrap().remove(0);
        let composer = inspector
            .subtree(&window.id, 64, 10_000)
            .unwrap()
            .into_iter()
            .map(|(_, e)| e)
            .find(|e| e.role == "Edit")
            .unwrap();
        let folder = std::env::temp_dir().join(format!("jevons-ui-clicks-{}", std::process::id()));
        let mut state = view(&folder);
        let context = ContextSnapshot {
            app: AppInfo {
                process_name: "slack.exe".into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: window.title.clone(),
                ..WindowInfo::default()
            },
            focused: Some(Focused {
                id: Some(composer.id.clone()),
                role: "Edit".into(),
                ..Focused::default()
            }),
            ..ContextSnapshot::default()
        };
        let top = browse::level(&inspector, &window.id).unwrap();
        let pane = top.elements[0].id.clone();
        state.interface = Some(InterfaceView {
            generation: 1,
            window: window.clone(),
            key: crate::agent::window_key(&context),
            levels: [(window.id.clone(), top)].into(),
            open: [window.id.clone()].into(),
            ..InterfaceView::default()
        });
        state.context = Some(context);
        let view = Arc::new(Mutex::new(state));
        let (commands, mut received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(browser_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(1400, 3000, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        let mut sent = || {
            let mut out = Vec::new();
            while let Ok(command) = received.try_recv() {
                out.push(command);
            }
            out
        };
        let rerender = |doc: &mut DioxusDocument| {
            doc.vdom.mark_dirty(ScopeId::APP);
            doc.poll(None);
        };
        sent();

        // The chevron opens the pane, and nothing is selected.
        click(&mut doc, &format!("[data-toggle=\"{pane}\"]"));
        let got = sent();
        assert!(
            matches!(got.as_slice(), [Command::InterfaceOpen { id, all: false }] if *id == pane),
            "{got:?}"
        );
        // The agent reads it; the same chevron now closes it.
        {
            let mut state = view.lock().unwrap();
            let browser = state.interface.as_mut().unwrap();
            browser
                .levels
                .insert(pane.clone(), browse::level(&inspector, &pane).unwrap());
            browser.open.insert(pane.clone());
        }
        rerender(&mut doc);
        click(&mut doc, &format!("[data-toggle=\"{pane}\"]"));
        let got = sent();
        assert!(
            matches!(got.as_slice(), [Command::InterfaceClose(id)] if *id == pane),
            "{got:?}"
        );
        // The label selects, and opens nothing.
        click(&mut doc, &format!("[data-row=\"{pane}\"] .iface-label"));
        let got = sent();
        assert!(
            matches!(got.as_slice(), [Command::InterfaceSelect(e)] if e.id == pane),
            "{got:?}"
        );

        // A level read in part ends with a button that reads more of it.
        view.lock()
            .unwrap()
            .interface
            .as_mut()
            .unwrap()
            .levels
            .get_mut(&pane)
            .unwrap()
            .total += 300;
        rerender(&mut doc);
        let text = doc.root_element().text_content();
        assert!(text.contains("more"), "{text}");
        click(&mut doc, &format!("[data-more=\"{pane}\"]"));
        let got = sent();
        assert!(
            matches!(got.as_slice(), [Command::InterfaceMore(id)] if *id == pane),
            "{got:?}"
        );

        // Expand to a level, collapse all, and show the focused element.
        click(&mut doc, "[data-level=\"3\"]");
        click(&mut doc, "[data-action=\"collapse\"]");
        click(&mut doc, "[data-action=\"show-focused\"]");
        let got = sent();
        assert!(
            matches!(
                got.as_slice(),
                [
                    Command::InterfaceExpand(3),
                    Command::InterfaceCollapse,
                    Command::InterfaceFocus
                ]
            ),
            "{got:?}"
        );

        // A search's matches list where they are; choosing one reveals it.
        let found = browse::search(
            &inspector,
            &window.id,
            "release notes",
            5_000,
            std::time::Duration::from_secs(2),
        )
        .unwrap();
        let ids = found.hits[0].ids();
        view.lock().unwrap().interface.as_mut().unwrap().search = Some(InterfaceSearch {
            query: "release notes".into(),
            found: Some(Ok(found)),
        });
        rerender(&mut doc);
        let text = doc.root_element().text_content();
        assert!(text.contains("2 matches"), "{text}");
        click(&mut doc, ".iface-result");
        let got = sent();
        assert!(
            matches!(got.as_slice(), [Command::InterfaceReveal(path)] if *path == ids),
            "{got:?}"
        );

        // With the whole window open the box scrolls, and a revealed row comes into view.
        {
            let mut state = view.lock().unwrap();
            let browser = state.interface.as_mut().unwrap();
            let all = browse::open_below(&inspector, &window.id, 64, 10_000, &Default::default())
                .unwrap();
            for (parent, level) in all.levels {
                browser.open.insert(parent.clone());
                browser.levels.insert(parent, level);
            }
        }
        rerender(&mut doc);
        let last = ids.last().unwrap();
        assert!(interface::scroll_into_view(&mut doc, last));
        let tree = doc.query_selector(".iface-tree").unwrap().unwrap();
        let row = doc
            .query_selector(&format!("[data-row=\"{last}\"]"))
            .unwrap()
            .unwrap();
        let (scroll, height) = {
            let tree = doc.get_node(tree).unwrap();
            (
                tree.scroll_offset.y,
                f64::from(tree.final_layout.size.height),
            )
        };
        assert!(scroll > 0.0, "the box scrolled");
        let top = f64::from(
            doc.get_node(row).unwrap().absolute_position(0.0, 0.0).y
                - doc.get_node(tree).unwrap().absolute_position(0.0, 0.0).y,
        );
        assert!(
            top >= scroll && top <= scroll + height,
            "{top} in {scroll}+{height}"
        );
        std::fs::remove_dir_all(&folder).ok();
    }
}
