# jevons-desktop

## Purpose

jevons-desktop is the tray app for context-aware dictation and small automations. A hotkey starts
a take in any application: the app reads which application and field are in front, listens on the
microphone, and runs the take through the flow tree, which types the text where the user was,
answers in a bubble by the tray icon, or calls a tool. It records demonstrations and runs the
automations written from them. It runs the models in-process through jevons-api, or uses a jevons
server elsewhere. The inspector window shows the live context, the takes, the flow tree, the
machines, the settings and the models. The binary also runs headless: one take from audio or
text, checks of a flows folder or an automations library, XPath queries, dry runs, a settings
reset and history clears.

## Scope

jevons-desktop owns:

- the agent thread: hotkey gestures, takes, confirmations in the bubble, recordings, automation
  runs from the tray, the settings and history actions, reloading the flows folder and library;
- the tray thread: the icon, its menu and the global hotkeys;
- the runtime thread: the embedded jevons-api (loaded models and its private listener), the other
  providers, each capability's route to one of them, and the forwarder's listener;
- the inspector window and the feedback bubble, on dioxus-native (Blitz);
- the platform layers for each OS: context, interface reads and actions, text input, the
  recorder, the microphone, and the keyboard hook for held hotkeys;
- the command line.

It leaves to other crates:

- `jevons-desktop-core`: the flow tree, machines, the take pipeline, paste safety, tools and
  agents, automations, recording, the settings folder and its repository, history clears, the
  API client, the catalog and downloads, and the tray icon's frames;
- `jevons-api`: the API it embeds (`load` and `serve`), with its routes and wire formats.

The platform code goes through safe wrapper crates only, and the crate forbids `unsafe`.

## Capabilities

Each file numbers its own requirements.

| Spec | Covers |
| --- | --- |
| [app](app.md) | The tray and its menu, hotkey gestures, takes in the app, confirmations, recordings and automation runs from the tray, settings, reset and history actions, reloading |
| [settings](settings.md) | The two settings files, their defaults, the folder's git repository, the reset and history clears |
| [runtime](runtime.md) | The embedded runtime and its listener, the other providers, each capability's route, the runtime's status |
| [ui](ui.md) | The inspector's tabs (Context, Takes, Flows, Machines, Settings, Models) and the feedback bubble |
| [platform](platform.md) | Context and text input per OS, the microphone, the keyboard hook, interface actions and the recorder on Windows |
| [cli](cli.md) | The command line: headless takes, checks, queries, dry runs, reset and clears |

## Threads

| Thread | Owns |
| --- | --- |
| main | the inspector window and the bubble (winit, Blitz) |
| tray | the tao event loop: the tray icon, its menu, the global hotkeys, the icon's animation |
| agent | gestures, the context, the microphone, takes (on its Tokio workers) |
| runtime | the loaded models and the API listener, rebound without reloading |
