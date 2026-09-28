//! The inspector and settings window, on dioxus-native (Blitz): the live context and why a
//! profile matches, the recent takes, the profiles, the settings and the models. Closing it hides
//! it; the tray reopens it.
//!
//! The window lives on the main thread's winit loop. The agent and background work call [`wake`],
//! which re-renders from the shared [`View`](crate::agent::View).

mod app;
mod components;
mod context;
mod models;
mod profiles;
mod settings;
mod takes;

use crate::agent::{Command, SharedView};
use anyrender_vello::{VelloRendererOptions, VelloWindowRenderer};
use blitz_shell::{BlitzApplication, BlitzShellEvent, WindowConfig};
use dioxus::prelude::*;
use dioxus_native::{DioxusDocument, DocumentConfig};
use std::sync::{Mutex, OnceLock};
use tokio::sync::mpsc::UnboundedSender;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

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
    let mut vdom = VirtualDom::new(app::App);
    vdom.insert_any_root_context(Box::new(Ctx {
        view: view.clone(),
        commands,
    }));
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
    let mut shell = Shell { inner, view };
    event_loop.run_app(&mut shell)?;
    Ok(())
}

struct Shell {
    inner: BlitzApplication<VelloWindowRenderer>,
    view: SharedView,
}

impl Shell {
    fn refresh(&mut self, event_loop: &ActiveEventLoop) {
        let (quit, show) = {
            let mut view = self.view.lock().expect("the view lock");
            (view.quit, std::mem::take(&mut view.show_window))
        };
        if quit {
            event_loop.exit();
            return;
        }
        for window in self.inner.windows.values_mut() {
            if show {
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
        self.refresh(event_loop);
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.suspended(event_loop);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
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
