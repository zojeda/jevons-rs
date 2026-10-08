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
mod machines;
mod markdown;
pub(crate) use markdown::plain as plain_text;
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

    /// Another tab was picked: the machine picked in the tray menu no longer shows, as a
    /// machine picked in the Machines tab is forgotten with it.
    pub fn leave_machine(&self) {
        let mut view = self.view.lock().expect("the view lock");
        view.machines_tab = false;
        view.open_machine = None;
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
        bubble_turn: None,
        bubble_follow: true,
        bubble_end: 0.0,
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
    /// The window and take whose text the bubble shows: a new one follows its newest text
    /// again.
    bubble_turn: Option<(WindowId, u64)>,
    /// Whether the bubble follows its newest text: until the user scrolls up, and again once
    /// they are back at the end.
    bubble_follow: bool,
    /// How far the bubble's text could scroll when last rendered, to notice more arriving.
    bubble_end: f64,
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
        let (quit, show, bubble, clicks, turn, jump) = {
            let mut view = self.view.lock().expect("the view lock");
            // A call waiting for confirmation shows even with live feedback off.
            let asking = view.feedback.as_ref().is_some_and(|f| f.confirm.is_some());
            let message = view.feedback.as_ref().is_some_and(|f| f.message);
            // An answer and a task's conversation are for reading: they show even with live
            // feedback off.
            let answer = view.feedback.as_ref().is_some_and(|f| f.reading());
            let bubble = view.feedback.is_some()
                && (view.config.dictation.live_feedback || asking || message || answer);
            // An answer takes the wheel as soon as it streams in, and clicks once it is done.
            let clicks = view
                .feedback
                .as_ref()
                .is_some_and(|f| f.confirm.is_some() || f.reading());
            // The take whose text the bubble shows, and whether its button asked for the
            // newest text.
            let turn = view
                .feedback
                .as_ref()
                .filter(|f| f.reading() && !asking)
                .map(|f| f.take);
            let jump = std::mem::take(&mut view.bubble.jump);
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
                turn,
                jump,
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
            // The bubble's text follows its newest part, as a chat does, until the user scrolls
            // up; then a button goes back to it, and says when more has arrived.
            if Some(*id) == self.bubble
                && let Some(take) = turn
                && let Some((_, end)) = bubble::scroll(window.downcast_doc_mut::<DioxusDocument>())
            {
                if jump || self.bubble_turn != Some((*id, take)) {
                    self.bubble_turn = Some((*id, take));
                    self.bubble_follow = true;
                }
                let more = end > self.bubble_end + 0.5;
                self.bubble_end = end;
                let changed = {
                    let mut view = self.view.lock().expect("the view lock");
                    let scroll = crate::agent::BubbleScroll {
                        away: !self.bubble_follow,
                        fresh: !self.bubble_follow && (view.bubble.fresh || more),
                        jump: false,
                    };
                    let changed = view.bubble != scroll;
                    view.bubble = scroll;
                    changed
                };
                if changed {
                    // The button shows, hides or lights up.
                    let doc = window.downcast_doc_mut::<DioxusDocument>();
                    doc.vdom.mark_dirty(ScopeId::APP);
                    window.poll();
                }
                if self.bubble_follow {
                    bubble::scroll_to_newest(window.downcast_doc_mut::<DioxusDocument>());
                }
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
        let wheel = Some(id) == self.bubble && matches!(event, WindowEvent::MouseWheel { .. });
        self.inner.window_event(event_loop, id, event);
        // The wheel moved the bubble's text: it follows the newest part only from its end.
        if wheel
            && let Some(window) = self.inner.windows.get_mut(&id)
            && let Some((at, end)) = bubble::scroll(window.downcast_doc_mut::<DioxusDocument>())
        {
            let follow = at >= end - 2.0;
            if follow != self.bubble_follow {
                self.bubble_follow = follow;
                self.refresh(event_loop);
            }
        }
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
    use crate::config::DesktopConfig;
    use blitz_dom::Document as _;
    use jevons_desktop_core::context::{AppInfo, ContextSnapshot, Element as Focused, WindowInfo};
    use jevons_desktop_core::platform::{Action, DeliveryMethod, DeliveryOutcome};
    use jevons_desktop_server::flow::spec::Output;
    use jevons_desktop_server::flow::walk::{self, Leaf};
    use jevons_desktop_server::flow::{Catalog, FlowTree, defaults};
    use jevons_desktop_server::pipeline::{Trace, TranscriptionPath};
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
            5 => rsx! { machines::MachinesPage { rev: 5 } },
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
                value: None,
                node: "_actions/insert".into(),
                text: "Hello world.".into(),
                output: Output::Target,
                action: Action::Insert,
                delivery: DeliveryMethod::Paste,
            }),
            calls: Vec::new(),
            machine: Vec::new(),
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
                    kind: jevons_desktop_server::pipeline::StageKind::Deciding,
                    label: "what to do".into(),
                    choices: vec!["assistant".into(), "dictation".into()],
                    detail: String::new(),
                    chosen: None,
                    ok: None,
                });
            }),
            Box::new(|f| f.frame += 5),
            Box::new(|f| {
                let stage = f.stages.last_mut().unwrap();
                stage.chosen = Some("dictation".into());
                stage.detail = "0.92".into();
                stage.ok = Some(true);
                f.stages.push(crate::agent::StageView {
                    kind: jevons_desktop_server::pipeline::StageKind::Writing,
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
            text.contains("what to do") && text.contains("dictation") && text.contains("0.92"),
            "{text}"
        );
        assert!(text.contains("Inserted"), "{text}");
        // A tool call waits, then the take answers in the bubble.
        view.lock().unwrap().feedback.as_mut().unwrap().confirm = Some(crate::agent::PendingCall {
            question: "Run notes:create_note?".into(),
            details: "{\"title\": \"Launch\"}".into(),
            action: "Run".into(),
        });
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        let text = doc.root_element().text_content();
        assert!(
            text.contains("Run notes:create_note?")
                && text.contains("Cancel (Esc)")
                && text.contains("Run (Enter)"),
            "{text}"
        );
        // The app's own questions say what they do.
        view.lock().unwrap().feedback.as_mut().unwrap().confirm = Some(crate::agent::PendingCall {
            question: "Clear the take traces?".into(),
            details: "Removes the take traces in C:\\Users\\me\\jevons\\traces".into(),
            action: "Clear".into(),
        });
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        let text = doc.root_element().text_content();
        assert!(
            text.contains("Clear the take traces?")
                && text.contains("Clear (Enter)")
                && text.contains(r"C:\Users\me\jevons\traces")
                && !text.contains("Run"),
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
                let dictate = flows.find("dictation/dictate").unwrap();
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
        for a in 0..=6 {
            for b in 0..=6 {
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
    /// The workbench with ask's `slack_messages` chosen, as the "Edit" of a reading picks it.
    fn workbench_root() -> Element {
        let rev = REV.fetch_add(1, Ordering::Relaxed);
        let chosen = use_signal(|| Some(workbench::key("assistant/ask", "slack_messages")));
        rsx! { workbench::Workbench { rev, chosen } }
    }

    #[test]
    fn the_extract_workbench_loads_the_chosen_extract_and_shows_a_trial_in_blitz() {
        use jevons_desktop_server::flow::extract::{Extracted, Trial};
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
        assert!(html.contains("Save to assistant/ask/decide.toml"), "{html}");
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
        assert!(text.contains("This tree shows slack.exe"), "{text}");
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
        doc.resolve(0.0);
        let node = doc
            .query_selector(selector)
            .unwrap()
            .unwrap_or_else(|| panic!("nothing matches {selector}"));
        click_node(doc, node, selector);
    }

    /// Clicks the first element matching `selector` whose text holds `text`.
    fn click_text(doc: &mut DioxusDocument, selector: &str, text: &str) {
        doc.resolve(0.0);
        let node = doc
            .query_selector_all(selector)
            .unwrap()
            .into_iter()
            .find(|n| doc.get_node(*n).unwrap().text_content().contains(text))
            .unwrap_or_else(|| panic!("no {selector} says {text}"));
        click_node(doc, node, selector);
    }

    fn click_node(doc: &mut DioxusDocument, node: usize, selector: &str) {
        use blitz_traits::events::{
            BlitzMouseButtonEvent, MouseEventButton, MouseEventButtons, UiEvent,
        };
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

    fn app_root() -> Element {
        rsx! { app::App {} }
    }

    #[test]
    fn the_tabs_take_clicks_while_the_page_is_scrolled() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let folder = std::env::temp_dir().join(format!("jevons-ui-tabs-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(app_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(1000, 600, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        doc.resolve(0.0);
        // The page scrolled far down: its content now lies under the tab bar.
        let page = doc.query_selector(".page").unwrap().unwrap();
        doc.get_node_mut(page).unwrap().scroll_offset = blitz_dom::Point { x: 0.0, y: 400.0 };
        click(&mut doc, ".dx-tabs-trigger:nth-child(2)");
        let active = doc
            .query_selector(".dx-tabs-trigger[data-state=\"active\"]")
            .unwrap()
            .unwrap();
        assert_eq!(doc.get_node(active).unwrap().text_content(), "Takes");
    }

    #[test]
    fn the_settings_say_automations_are_coming_soon_until_they_are_turned_on() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let folder = std::env::temp_dir().join(format!("jevons-ui-soon-{}", std::process::id()));
        let settings = |on: bool| {
            let mut state = view(&folder);
            state.config.automation.enabled = on;
            let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
            let mut vdom = VirtualDom::new(app::App);
            vdom.insert_any_root_context(Box::new(Ctx {
                view: Arc::new(Mutex::new(state)),
                commands,
            }));
            let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
            doc.add_user_agent_stylesheet(include_str!("style.css"));
            doc.set_viewport(Viewport::new(1000, 4000, 1.0, ColorScheme::Dark));
            doc.initial_build();
            doc.poll(None);
            click_text(&mut doc, ".topbar .dx-tabs-trigger", "Settings");
            doc
        };
        // Off, as the defaults are: the card says so and how to try them, and offers no hotkey.
        let doc = settings(false);
        let card = texts(&doc, ".dx-card[data-soon=\"true\"]").join(" ");
        assert!(
            card.contains("Automations")
                && card.contains("Coming soon")
                && card.contains("enabled = true")
                && !card.contains("Starts recording"),
            "{card}"
        );
        // Turned on in the settings file: the record hotkey is there to set.
        let doc = settings(true);
        assert!(texts(&doc, ".dx-card[data-soon=\"true\"]").is_empty());
        let card = texts(&doc, ".dx-card[data-soon=\"false\"]").join(" ");
        assert!(
            card.contains("Starts recording") && !card.contains("Coming soon"),
            "{card}"
        );
    }

    #[test]
    fn edited_settings_mark_the_page_and_save_from_a_bar_that_shows_only_then() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let folder = std::env::temp_dir().join(format!("jevons-ui-dirty-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        let (commands, mut received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(app_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        // Tall enough that the whole settings page is in view: what is scrolled out takes no clicks.
        doc.set_viewport(Viewport::new(1000, 4000, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        let shows =
            |doc: &DioxusDocument, selector: &str| doc.query_selector(selector).unwrap().is_some();
        click_text(&mut doc, ".dx-tabs-trigger", "Settings");
        assert!(!shows(&doc, ".save-bar") && !shows(&doc, ".tab-dot"));
        // A change marks the page and the tab, and the save bar appears.
        click_text(&mut doc, ".switch-row", "Include the clipboard");
        assert!(shows(&doc, ".save-bar") && shows(&doc, ".tab-dot"));
        assert!(shows(&doc, ".page[data-dirty=\"true\"]"));
        assert!(
            doc.root_element()
                .text_content()
                .contains("Unsaved changes")
        );
        // Revert drops it.
        click_text(&mut doc, ".save-bar .dx-button", "Revert");
        assert!(!shows(&doc, ".save-bar") && !shows(&doc, ".tab-dot"));
        // Changed again and applied: the agent gets the settings, and the bar goes.
        click_text(&mut doc, ".switch-row", "Include the clipboard");
        click_text(&mut doc, ".save-bar .dx-button", "Apply and save");
        let mut applied = None;
        while let Ok(command) = received.try_recv() {
            if let crate::agent::Command::Apply(config) = command {
                applied = Some(config);
            }
        }
        let applied = applied.expect("Apply sent");
        assert!(applied.privacy.read_clipboard);
        assert_eq!(applied.models, view.lock().unwrap().config.models);
        // The agent saves them; the window follows and nothing is pending.
        view.lock().unwrap().config = *applied;
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        assert!(!shows(&doc, ".save-bar") && !shows(&doc, ".tab-dot"));
    }

    fn flows_root() -> Element {
        rsx! { flows::FlowsPage { rev: REV.fetch_add(1, Ordering::Relaxed) } }
    }

    #[test]
    fn the_flow_tree_nests_branches_folds_them_and_shows_a_node_in_full() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let folder = std::env::temp_dir().join(format!("jevons-ui-flows-{}", std::process::id()));
        let mut state = view(&folder);
        state.flows = flows();
        let context = state.context.clone().unwrap();
        state.route = walk::preview(&state.flows, &context, state.flows.root());
        let view = Arc::new(Mutex::new(state));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(flows_root);
        vdom.insert_any_root_context(Box::new(Ctx { view, commands }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(1200, 6000, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        let names = |doc: &DioxusDocument, selector: &str| -> Vec<String> {
            doc.query_selector_all(selector)
                .unwrap()
                .into_iter()
                .map(|n| doc.get_node(n).unwrap().text_content())
                .collect()
        };
        // dictate's branches sit two levels down, under the root and under dictate.
        let nested = names(
            &doc,
            ".flow-children .flow-children > .flow-node > .flow-row .flow-name",
        );
        assert!(nested.contains(&"terminal".to_string()), "{nested:?}");
        // Shared actions say where they come from.
        assert!(
            names(&doc, ".flow-shared")
                .iter()
                .any(|s| s == "shared from _actions")
        );
        // dictate's row: its toggle folds it, its label selects it.
        doc.resolve(0.0);
        let dictate = doc
            .query_selector_all(".flow-row")
            .unwrap()
            .into_iter()
            .find(|n| {
                let row = doc.get_node(*n).unwrap();
                doc.get_node(row.children[1])
                    .unwrap()
                    .text_content()
                    .contains("decidedictate")
            })
            .unwrap();
        let (toggle, label) = {
            let row = doc.get_node(dictate).unwrap();
            (row.children[0], row.children[1])
        };
        click_node(&mut doc, label, ".flow-label");
        doc.resolve(0.0);
        click_node(&mut doc, toggle, ".flow-toggle");
        let nested = names(
            &doc,
            ".flow-children .flow-children > .flow-node > .flow-row .flow-name",
        );
        assert!(!nested.contains(&"terminal".to_string()), "{nested:?}");
        // Selecting dictate showed it in full.
        let text = doc.root_element().text_content();
        assert!(
            text.contains("Applies when") && text.contains("dictation/dictate/decide.toml"),
            "{text}"
        );
    }

    fn machines_root() -> Element {
        rsx! { machines::MachinesPage { rev: REV.fetch_add(1, Ordering::Relaxed) } }
    }

    /// The Machines page over `view`, laid out on a wide dark viewport.
    fn machines_doc(view: View) -> DioxusDocument {
        machines_doc_with(view).0
    }

    /// The Machines tab over `view`, and what its buttons ask the agent for.
    fn machines_doc_with(
        view: View,
    ) -> (
        DioxusDocument,
        tokio::sync::mpsc::UnboundedReceiver<crate::agent::Command>,
    ) {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let (commands, received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(machines_root);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: Arc::new(Mutex::new(view)),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(1200, 3000, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        doc.resolve(0.0);
        (doc, received)
    }

    fn texts(doc: &DioxusDocument, selector: &str) -> Vec<String> {
        doc.query_selector_all(selector)
            .unwrap()
            .into_iter()
            .map(|n| doc.get_node(n).unwrap().text_content())
            .collect()
    }

    #[test]
    fn the_machines_page_draws_the_built_in_root_machine() {
        let folder = std::env::temp_dir().join(format!("jevons-ui-fsm-{}", std::process::id()));
        let doc = machines_doc(view(&folder));
        let mut names = texts(&doc, ".fsm-state .fsm-name");
        names.sort();
        assert_eq!(names, ["assistant", "automations", "dictation", "idle"]);
        // idle has no folder: it waits; the others' work is an agent, a machine of its own.
        let works = texts(&doc, ".fsm-state .fsm-work");
        assert!(works.contains(&"waits".to_string()), "{works:?}");
        assert!(works.contains(&"machine".to_string()), "{works:?}");
        // The start dot, an arrowhead per edge, and the [else] label.
        assert!(doc.query_selector(".fsm-lines circle").unwrap().is_some());
        assert!(doc.query_selector_all(".fsm-lines polygon").unwrap().len() >= 7);
        assert!(texts(&doc, ".fsm-label").iter().any(|l| l == "said [else]"));
        // Each edge says what decides it: the agents' ends are the event alone, and what the
        // user said is for rules and then the model. The legend names the colours.
        let decided = |by: &str| {
            let selector = format!(".fsm-lines g[data-by=\"{by}\"]");
            doc.query_selector_all(&selector).unwrap().len()
        };
        assert_eq!(
            (
                decided("none"),
                decided("event"),
                decided("rules_then_model")
            ),
            (1, 3, 3)
        );
        assert_eq!(
            texts(&doc, ".fsm-legend-item"),
            ["the event", "rules", "rules, then the model", "the model"]
        );
        // States are boxes where the layout put them, not collapsed.
        let state = doc.query_selector(".fsm-state").unwrap().unwrap();
        let size = doc.get_node(state).unwrap().final_layout.size;
        assert!(size.width > 150.0 && size.height > 40.0, "{size:?}");
        // Nothing runs yet: no state is current and there is no task to cancel.
        assert!(
            doc.query_selector(".fsm-state[data-current=\"true\"]")
                .unwrap()
                .is_none()
        );
        let text = doc.root_element().text_content();
        assert!(text.contains("not started"), "{text}");
    }

    /// An agent whose task waits, as one take left them: the root handed the take to the
    /// agent, which started the task. No model is asked.
    fn a_task_that_waits() -> (
        jevons_desktop_server::pipeline::Env,
        Arc<jevons_desktop_server::flow::machine::runtime::Runtime>,
        tokio::sync::mpsc::UnboundedSender<jevons_desktop_server::pipeline::Update>,
    ) {
        use jevons_desktop_server::flow::Memory;
        use jevons_desktop_server::flow::machine::runtime::Runtime as Machines;
        use jevons_desktop_server::pipeline::{Env, Settings, TakeStart};
        let tree = Arc::new(FlowTree::load(
            &Memory::new(
                "test",
                [
                    ("root.toml", ""),
                    (
                        "root.fsm",
                        "fsm App {\n[*] --> idle\nidle --> helper : said\nhelper --> idle\n}",
                    ),
                    ("helper/agent.toml", "description = \"Helps\""),
                    (
                        "helper/agent.fsm",
                        "fsm Helper {\n[*] --> idle\nidle --> task : said\ntask --> idle\n}",
                    ),
                    ("helper/task/task.toml", "description = \"A task\""),
                    (
                        "helper/task/task.fsm",
                        "fsm Task {\n[*] --> waiting\nstate waiting: \"Waiting for the go\"\nwaiting --> [*] : said [the user says go]\n}",
                    ),
                ],
            ),
            &Catalog::default(),
        ));
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let machines = Arc::new(Machines::new());
        let env = Env {
            routes: jevons_desktop_server::client::Routes::default(),
            flows: tree.clone(),
            settings: Settings::default(),
            desk: std::sync::Arc::new(jevons_desktop_protocol::desk::Nobody),
            investigator: None,
            tools: None,
            machines: machines.clone(),
        };
        let start = TakeStart {
            id: 7,
            context: ContextSnapshot::default(),
            entry: None,
        };
        let mut trace = Trace::new(&start);
        trace.transcript = "start the task".into();
        let (updates, _) = tokio::sync::mpsc::unbounded_channel();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(machines.take(&env, &start, None, &updates, &mut trace));
        assert_eq!(
            machines.view().path(),
            "helper › task › waiting",
            "{:?}",
            trace.notes
        );
        (env, machines, updates)
    }

    #[test]
    fn a_running_task_shows_under_its_agent_with_its_state_and_can_be_cancelled() {
        use crate::agent::Command;
        use jevons_desktop_server::pipeline::TakeStart;
        let (env, machines, updates) = a_task_that_waits();
        let folder = std::env::temp_dir().join(format!("jevons-ui-task-{}", std::process::id()));
        let mut state = view(&folder);
        state.flows = env.flows.clone();
        state.machines = machines.clone();
        let (mut doc, _) = machines_doc_with(state);
        // What runs: the root, the agent, and the agent's task under it.
        assert_eq!(texts(&doc, ".fsm-run-name"), ["/", "helper", "task-1"]);
        assert_eq!(
            texts(&doc, ".fsm-run[data-level=\"task\"] .fsm-path"),
            ["waiting"]
        );
        // The task the take reached shows first, at its current state.
        assert_eq!(
            texts(&doc, ".fsm-run[data-showing=\"true\"] .fsm-run-name"),
            ["task-1"]
        );
        assert_eq!(
            texts(&doc, ".fsm-state[data-current=\"true\"] .fsm-name"),
            ["waiting"]
        );
        let text = doc.root_element().text_content();
        assert!(
            text.contains("helper › task › waiting") && text.contains("waiting for said"),
            "{text}"
        );
        // The history: the root's hand-off, the agent's move, then the task's start.
        let steps = texts(&doc, ".fsm-step-move");
        assert!(steps.contains(&"idle → helper".to_string()), "{steps:?}");
        assert!(steps.contains(&"idle → task".to_string()), "{steps:?}");
        assert!(steps.contains(&"[*] → waiting".to_string()), "{steps:?}");
        // Its latest transition has no edge (a start), so no label is lit.
        assert!(
            doc.query_selector(".fsm-label[data-hot=\"true\"]")
                .unwrap()
                .is_none()
        );
        // Selecting the state shows it in full.
        click_text(&mut doc, ".fsm-state", "waiting");
        let card = texts(&doc, ".fsm-detail").join(" ");
        assert!(
            card.contains("Waiting for the go")
                && card.contains("No folder")
                && card.contains("[the user says go]")
                && card.contains("Here now"),
            "{card}"
        );
        // Selecting the agent shows its diagram, at the state it waits in.
        click_text(&mut doc, ".fsm-run", "helper");
        assert_eq!(
            texts(&doc, ".fsm-state[data-current=\"true\"] .fsm-name"),
            ["idle"]
        );
        // What is said next has two takers, the agent's own transition and its task, and no
        // model to say which: the agent stays, and its diagram offers the candidates.
        let start = TakeStart {
            id: 8,
            context: ContextSnapshot::default(),
            entry: None,
        };
        let mut trace = Trace::new(&start);
        trace.transcript = "go on".into();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(machines.take(&env, &start, None, &updates, &mut trace));
        let helper = machines.view().stack[1].id;
        assert!(machines.view().running(helper).unwrap().unsure.is_some());
        let mut state = view(&folder);
        state.flows = env.flows.clone();
        state.machines = machines.clone();
        let (mut doc, mut commands) = machines_doc_with(state);
        click_text(&mut doc, ".fsm-run", "helper");
        let unsure = texts(&doc, ".fsm-unsure").join(" ");
        assert!(
            unsure.contains("Unsure what you meant") && unsure.contains("“go on”"),
            "{unsure}"
        );
        assert_eq!(texts(&doc, ".fsm-answer"), ["task", "task-1"]);
        click_text(&mut doc, ".fsm-answer", "task-1");
        let mut answers = Vec::new();
        while let Ok(command) = commands.try_recv() {
            if let Command::AnswerDecision { instance, label } = command {
                answers.push((instance, label));
            }
        }
        assert_eq!(answers, [(helper, "task-1".to_string())]);
        // A task has its own Cancel; the root and the agent have none.
        assert_eq!(texts(&doc, ".fsm-run button"), ["Cancel"]);
        click(&mut doc, ".fsm-run button");
        let task = machines.view().stack[2].id;
        let mut asked = Vec::new();
        while let Ok(command) = commands.try_recv() {
            if let Command::CancelOneTask(id) = command {
                asked.push(id);
            }
        }
        assert_eq!(asked, [task]);
    }

    #[test]
    fn a_machine_picked_in_the_tray_shows_in_the_machines_tab() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let (env, machines, _updates) = a_task_that_waits();
        let (helper, task) = (machines.view().stack[1].id, machines.view().stack[2].id);
        let folder = std::env::temp_dir().join(format!("jevons-ui-pick-{}", std::process::id()));
        let state = |open: Option<u64>| {
            let mut state = view(&folder);
            state.flows = env.flows.clone();
            state.machines = machines.clone();
            state.open_machine = open;
            state
        };
        let showing =
            |doc: &DioxusDocument| texts(doc, ".fsm-run[data-showing=\"true\"] .fsm-run-name");
        let current =
            |doc: &DioxusDocument| texts(doc, ".fsm-state[data-current=\"true\"] .fsm-name");
        // The agent was picked: its diagram shows, at its state, not the task the take reached.
        let (mut doc, _) = machines_doc_with(state(Some(helper)));
        assert_eq!(
            (showing(&doc), current(&doc)),
            (vec!["helper".into()], vec!["idle".into()])
        );
        // Another machine picked in the tab shows from then on.
        click_text(&mut doc, ".fsm-run", "task-1");
        assert_eq!(
            (showing(&doc), current(&doc)),
            (vec!["task-1".into()], vec!["waiting".into()])
        );
        // A task picked shows; a machine that no longer runs leaves the tab as it is otherwise.
        let (doc, _) = machines_doc_with(state(Some(task)));
        assert_eq!(showing(&doc), ["task-1"]);
        let (doc, _) = machines_doc_with(state(Some(task + 100)));
        assert_eq!(showing(&doc), ["task-1"]);

        // The whole window: the Machines tab shows whichever tab was open (Context, at first),
        // and stays while machines are picked in it, until another tab is picked.
        let mut window = state(Some(helper));
        window.machines_tab = true;
        let shared = Arc::new(Mutex::new(window));
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        // The App is the root, as in the window: a wake renders it again.
        let mut vdom = VirtualDom::new(app::App);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: shared.clone(),
            commands,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(1200, 3000, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        doc.resolve(0.0);
        let tab =
            |doc: &DioxusDocument| texts(doc, ".topbar .dx-tabs-trigger[data-state=\"active\"]");
        assert_eq!(
            (tab(&doc), showing(&doc)),
            (vec!["Machines".into()], vec!["helper".into()])
        );
        click_text(&mut doc, ".fsm-run", "task-1");
        // The agent wakes the window, as it does on every take.
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        assert_eq!(
            (tab(&doc), showing(&doc)),
            (vec!["Machines".into()], vec!["task-1".into()])
        );
        // Context was the tab open before: picking it goes back to it.
        click_text(&mut doc, ".topbar .dx-tabs-trigger", "Context");
        assert_eq!(tab(&doc), ["Context"]);
        assert!(doc.query_selector(".fsm-run").unwrap().is_none());
        assert!(!shared.lock().unwrap().machines_tab);
        // The machine picked in the tray went with the tab, as one picked in the tab does.
        shared.lock().unwrap().open_machine = Some(helper);
        shared.lock().unwrap().machines_tab = true;
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        assert_eq!(showing(&doc), ["helper"]);
        click_text(&mut doc, ".topbar .dx-tabs-trigger", "Takes");
        assert_eq!(shared.lock().unwrap().open_machine, None);
        click_text(&mut doc, ".topbar .dx-tabs-trigger", "Machines");
        assert_eq!(
            (tab(&doc), showing(&doc)),
            (vec!["Machines".into()], vec!["task-1".into()])
        );
    }

    /// The route of a Slack reply through the built-in dictate branch, three decisions deep.
    fn route_root() -> Element {
        let tree = FlowTree::load(&defaults::builtin(), &Catalog::default());
        let context = ContextSnapshot {
            app: AppInfo {
                process_name: "slack.exe".into(),
                ..AppInfo::default()
            },
            focused: Some(Focused {
                role: "Edit".into(),
                name: "Reply to thread".into(),
                is_editable: true,
                selection: Some("hi".into()),
                ..Focused::default()
            }),
            ..ContextSnapshot::default()
        };
        let route = walk::preview(&tree, &context, tree.find("dictation/dictate").unwrap());
        rsx! { {context::route_card(&route, "A take's route")} }
    }

    #[test]
    fn a_route_nests_each_decision_under_the_branch_it_took() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let mut doc = DioxusDocument::new(VirtualDom::new(route_root), DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(1200, 3000, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        let names = |doc: &DioxusDocument, selector: &str| -> Vec<String> {
            doc.query_selector_all(selector)
                .unwrap()
                .into_iter()
                .map(|n| doc.get_node(n).unwrap().text_content())
                .collect()
        };
        // dictate → chat → thread, each a level deeper, on the chosen rows.
        let chosen = names(&doc, ".flow-row[data-route=\"true\"] .flow-name");
        assert_eq!(
            chosen,
            ["dictation/dictate", "chat", "thread"],
            "{chosen:?}"
        );
        let deep = names(
            &doc,
            ".flow-children .flow-children > .flow-node > .flow-row .flow-name",
        );
        assert!(deep.contains(&"thread".to_string()), "{deep:?}");
        // A branch not taken folds its rule checks until asked.
        assert!(doc.query_selector(".route-checks").unwrap().is_none());
        doc.resolve(0.0);
        let code = doc
            .query_selector_all(".flow-row[data-off=\"true\"]")
            .unwrap()
            .into_iter()
            .find(|n| doc.get_node(*n).unwrap().text_content().contains("code"))
            .map(|row| doc.get_node(row).unwrap().children[0])
            .unwrap();
        click_node(&mut doc, code, ".flow-toggle");
        let checks = names(&doc, ".route-checks");
        assert!(checks.iter().any(|c| c.contains("app")), "{checks:?}");
    }

    #[test]
    fn an_answer_turns_into_selectable_text_and_copies_as_plain_text_or_markdown() {
        use crate::agent::{BubbleAction, Command, Feedback};
        use blitz_traits::shell::{ColorScheme, Viewport};
        let folder = std::env::temp_dir().join(format!("jevons-ui-select-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        view.lock().unwrap().feedback = Some(Feedback {
            take: 1,
            answer: true,
            done: true,
            status: "Answered".into(),
            output: "The launch is **on Friday**.".into(),
            ..Feedback::default()
        });
        let (commands, mut received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(bubble::Bubble);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        vdom.insert_any_root_context(Box::new(bubble::Anchor {
            tail_x: 300.0,
            icon_below: true,
            width: bubble::ANSWER_SIZE.0,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("bubble.css"));
        doc.set_viewport(Viewport::new(
            bubble::ANSWER_SIZE.0 as u32,
            bubble::ANSWER_SIZE.1 as u32,
            1.0,
            ColorScheme::Dark,
        ));
        doc.initial_build();
        doc.poll(None);
        assert!(doc.query_selector("textarea").unwrap().is_none());
        click_text(&mut doc, ".bubble-button", "Select text");
        doc.resolve(0.0);
        let field = doc
            .query_selector("textarea")
            .unwrap()
            .expect("a text field");
        let text = doc
            .get_node(field)
            .unwrap()
            .element_data()
            .unwrap()
            .text_input_data()
            .unwrap()
            .editor
            .text()
            .to_string();
        assert!(
            text.contains("**on Friday**"),
            "the Markdown as written: {text}"
        );
        click_text(&mut doc, ".bubble-button", "Copy raw");
        click(&mut doc, ".bubble-button[data-primary=\"true\"]");
        let mut copies = Vec::new();
        while let Ok(command) = received.try_recv() {
            if let Command::Bubble(BubbleAction::Copy { raw }) = command {
                copies.push(raw);
            }
        }
        assert_eq!(copies, [true, false]);
        // Back to the formatted answer.
        click_text(&mut doc, ".bubble-button", "Done selecting");
        assert!(doc.query_selector("textarea").unwrap().is_none());
    }

    #[test]
    fn a_waiting_task_s_bubble_shows_its_turns_and_goes_back_to_the_newest() {
        use crate::agent::{Feedback, Turn};
        use blitz_traits::shell::{ColorScheme, Viewport};
        let folder = std::env::temp_dir().join(format!("jevons-ui-turns-{}", std::process::id()));
        let view = Arc::new(Mutex::new(view(&folder)));
        // A long first answer, so the latest turn starts below what the bubble shows.
        let answer = (1..=40)
            .map(|i| format!("{i}. Result number {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        view.lock().unwrap().feedback = Some(Feedback {
            take: 2,
            task: true,
            done: true,
            state: "search › results".into(),
            heard: "Y eso otro".into(),
            status: "Search stayed at results: unsure".into(),
            turns: vec![Turn {
                heard: "Buscar máquinas de estados".into(),
                output: answer,
                answer: true,
                status: "Answer".into(),
                failed: false,
            }],
            ..Feedback::default()
        });
        let (commands, _received) = tokio::sync::mpsc::unbounded_channel();
        let mut vdom = VirtualDom::new(bubble::Bubble);
        vdom.insert_any_root_context(Box::new(Ctx {
            view: view.clone(),
            commands,
        }));
        vdom.insert_any_root_context(Box::new(bubble::Anchor {
            tail_x: 300.0,
            icon_below: true,
            width: bubble::ANSWER_SIZE.0,
        }));
        let mut doc = DioxusDocument::new(vdom, DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("bubble.css"));
        doc.set_viewport(Viewport::new(
            bubble::ANSWER_SIZE.0 as u32,
            bubble::ANSWER_SIZE.1 as u32,
            1.0,
            ColorScheme::Dark,
        ));
        doc.initial_build();
        doc.poll(None);
        doc.resolve(0.0);
        let shown = texts(&doc, ".bubble").join(" ");
        for expected in [
            "search › results",
            "Buscar máquinas de estados",
            "Result number 40",
            "Y eso otro",
            "Search stayed at results: unsure",
        ] {
            assert!(shown.contains(expected), "{expected:?} in {shown}");
        }
        // The earlier answer is still what Copy takes, so the answer's buttons show.
        for label in ["Close", "Select text", "Copy raw", "Copy"] {
            assert!(shown.contains(label), "{label:?} in {shown}");
        }
        // The text starts at its top and can scroll; following the newest part puts it at the
        // end.
        let (at, end) = bubble::scroll(&mut doc).expect("text to scroll");
        assert!(at == 0.0 && end > 100.0, "{at} of {end}");
        bubble::scroll_to_newest(&mut doc);
        assert_eq!(bubble::scroll(&mut doc), Some((end, end)));
        // At the end there is no button. Scrolled up, it shows, and says when more arrived.
        assert!(doc.query_selector(".bubble-newest").unwrap().is_none());
        view.lock().unwrap().bubble = crate::agent::BubbleScroll {
            away: true,
            fresh: true,
            jump: false,
        };
        doc.vdom.mark_dirty(ScopeId::APP);
        doc.poll(None);
        assert_eq!(texts(&doc, ".bubble-newest[data-fresh=\"true\"]"), ["New"]);
        // Pressing it asks the window for the newest text.
        click(&mut doc, ".bubble-newest");
        assert!(view.lock().unwrap().bubble.jump);
    }

    fn answer_root() -> Element {
        rsx! {
            div { class: "bubble-answer",
                {markdown::render(
                    "El texto incluye **mensajes y fotos** del caso.\n\n\
                     - La evidencia consiste en **capturas de pantalla**.\n- Otro `punto`.\n",
                )}
            }
        }
    }

    #[test]
    fn answers_keep_the_spaces_around_bold_text_and_list_items_flow_as_one_line() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let mut doc = DioxusDocument::new(VirtualDom::new(answer_root), DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("bubble.css"));
        doc.set_viewport(Viewport::new(500, 400, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        doc.resolve(0.0);
        // The text each block lays out as one run, as Blitz builds it.
        let runs: Vec<String> = doc
            .query_selector_all(".md-p, .md-item-body")
            .unwrap()
            .into_iter()
            .filter_map(|id| {
                let node = doc.get_node(id).unwrap();
                let layout = node.element_data()?.inline_layout_data.as_ref()?;
                Some(layout.text.clone())
            })
            .collect();
        assert_eq!(
            runs,
            [
                "El texto incluye mensajes y fotos del caso.",
                "La evidencia consiste en capturas de pantalla.",
                "Otro punto.",
            ]
        );
    }

    /// A text field, for the paint check below.
    fn field_root() -> Element {
        rsx! { input { class: "dx-input", value: "//TreeItem" } }
    }

    #[test]
    fn the_text_caret_is_painted_in_the_field_s_text_colour_on_the_dark_theme() {
        use blitz_traits::shell::{ColorScheme, Viewport};
        let mut doc = DioxusDocument::new(VirtualDom::new(field_root), DocumentConfig::default());
        doc.add_user_agent_stylesheet(include_str!("style.css"));
        doc.set_viewport(Viewport::new(800, 200, 1.0, ColorScheme::Dark));
        doc.initial_build();
        doc.poll(None);
        let field = doc.query_selector("input").unwrap().unwrap();
        assert!(doc.set_focus_to(field));
        doc.resolve(0.0);
        let mut painted = Painted::default();
        blitz_paint::paint_scene(&mut painted, &doc, 1.0, 800, 200);
        // The caret is the thin, tall fill of a focused field.
        let caret = painted
            .fills
            .iter()
            .find(|(r, _)| {
                r.width() > 0.5 && r.width() <= 2.0 && (8.0..=40.0).contains(&r.height())
            })
            .expect("a caret");
        assert_ne!(
            caret.1,
            [0, 0, 0, 255],
            "a black caret is invisible on the dark theme"
        );
        assert!(caret.1[..3].iter().all(|c| *c > 128), "{:?}", caret.1);
        // The caret (and the text it stands in) sits in the middle of the field, not at its top.
        let node = doc.get_node(field).unwrap();
        let (top, height) = (
            f64::from(node.absolute_position(0.0, 0.0).y),
            f64::from(node.final_layout.size.height),
        );
        let middle = (caret.0.y0 + caret.0.y1) / 2.0;
        assert!(
            (middle - (top + height / 2.0)).abs() <= 2.0,
            "caret {:?} in a field from {top} to {}",
            caret.0,
            top + height
        );
    }

    /// A scene that records what is filled where.
    #[derive(Default)]
    struct Painted {
        fills: Vec<(peniko::kurbo::Rect, [u8; 4])>,
    }

    impl anyrender::PaintScene for Painted {
        fn reset(&mut self) {}
        fn push_layer(
            &mut self,
            _blend: impl Into<peniko::BlendMode>,
            _alpha: f32,
            _transform: peniko::kurbo::Affine,
            _clip: &impl peniko::kurbo::Shape,
        ) {
        }
        fn pop_layer(&mut self) {}
        fn stroke<'a>(
            &mut self,
            _style: &peniko::kurbo::Stroke,
            _transform: peniko::kurbo::Affine,
            _brush: impl Into<anyrender::PaintRef<'a>>,
            _brush_transform: Option<peniko::kurbo::Affine>,
            _shape: &impl peniko::kurbo::Shape,
        ) {
        }
        fn fill<'a>(
            &mut self,
            _style: peniko::Fill,
            transform: peniko::kurbo::Affine,
            brush: impl Into<anyrender::PaintRef<'a>>,
            _brush_transform: Option<peniko::kurbo::Affine>,
            shape: &impl peniko::kurbo::Shape,
        ) {
            if let anyrender::PaintRef::Solid(color) = brush.into() {
                let rgba = color.to_rgba8();
                self.fills.push((
                    transform.transform_rect_bbox(shape.bounding_box()),
                    [rgba.r, rgba.g, rgba.b, rgba.a],
                ));
            }
        }
        fn draw_glyphs<'a, 's: 'a>(
            &'s mut self,
            _font: &'a peniko::FontData,
            _font_size: f32,
            _hint: bool,
            _normalized_coords: &'a [anyrender::NormalizedCoord],
            _style: impl Into<peniko::StyleRef<'a>>,
            _brush: impl Into<anyrender::PaintRef<'a>>,
            _brush_alpha: f32,
            _transform: peniko::kurbo::Affine,
            _glyph_transform: Option<peniko::kurbo::Affine>,
            _glyphs: impl Iterator<Item = anyrender::Glyph>,
        ) {
        }
        fn draw_box_shadow(
            &mut self,
            _transform: peniko::kurbo::Affine,
            _rect: peniko::kurbo::Rect,
            _brush: peniko::Color,
            _radius: f64,
            _std_dev: f64,
        ) {
        }
    }
}
