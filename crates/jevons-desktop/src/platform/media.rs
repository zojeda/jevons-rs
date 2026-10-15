//! The headphones' buttons on Windows. A Bluetooth headset's buttons never come as keys: Windows
//! hands them to the current media session, as it does to a music app. jevons keeps a paused
//! session of its own (a `MediaPlayer`'s transport controls, with its command manager off).
//! Windows makes it current when it starts, and again once the music that played pauses: Next
//! (a double-tap on most earbuds) and play or pause (a tap) reach it then. A tap goes on to the
//! last music app that was current, so the earbuds still play and pause it.

use super::HeadphoneButton;
use std::sync::{Arc, Mutex};
use windows::Foundation::TypedEventHandler;
use windows::Media::Control::{
    GlobalSystemMediaTransportControlsSession as MediaSession,
    GlobalSystemMediaTransportControlsSessionManager as Sessions,
};
use windows::Media::Playback::MediaPlayer;
use windows::Media::{
    MediaPlaybackStatus, MediaPlaybackType, SystemMediaTransportControls,
    SystemMediaTransportControlsButton, SystemMediaTransportControlsButtonPressedEventArgs,
};
use windows::core::{HSTRING, Ref};

/// jevons' media session, heard until dropped.
pub struct Headphones {
    /// The transport controls live as long as their player.
    _player: MediaPlayer,
    controls: SystemMediaTransportControls,
    pressed: i64,
    sessions: Sessions,
    changed: i64,
    /// The last current session that was not jevons' own.
    music: Arc<Mutex<Option<HSTRING>>>,
}

impl Headphones {
    pub fn start(on_button: Box<dyn Fn(HeadphoneButton) + Send + Sync>) -> Result<Self, String> {
        Self::open(on_button).map_err(|e| e.message())
    }

    fn open(on_button: Box<dyn Fn(HeadphoneButton) + Send + Sync>) -> windows::core::Result<Self> {
        let player = MediaPlayer::new()?;
        player.CommandManager()?.SetIsEnabled(false)?;
        let controls = player.SystemMediaTransportControls()?;
        controls.SetIsEnabled(true)?;
        controls.SetIsPlayEnabled(true)?;
        controls.SetIsPauseEnabled(true)?;
        controls.SetIsNextEnabled(true)?;
        controls.SetPlaybackStatus(MediaPlaybackStatus::Paused)?;
        let display = controls.DisplayUpdater()?;
        display.SetType(MediaPlaybackType::Music)?;
        let card = display.MusicProperties()?;
        card.SetTitle(&HSTRING::from("jevons"))?;
        card.SetArtist(&HSTRING::from("Double-tap to dictate"))?;
        display.Update()?;
        let pressed = controls.ButtonPressed(&TypedEventHandler::new(
            move |_, args: Ref<SystemMediaTransportControlsButtonPressedEventArgs>| {
                if let Some(button) = button(args.ok()?.Button()?) {
                    on_button(button);
                }
                Ok(())
            },
        ))?;
        let sessions = Sessions::RequestAsync()?.join()?;
        let music = Arc::new(Mutex::new(None));
        remember(&sessions, &music);
        let (watched, seen) = (sessions.clone(), music.clone());
        let changed = sessions.CurrentSessionChanged(&TypedEventHandler::new(move |_, _| {
            remember(&watched, &seen);
            Ok(())
        }))?;
        tracing::info!(sessions = %listed(&sessions), "Listening to the headphones' buttons");
        Ok(Self {
            _player: player,
            controls,
            pressed,
            sessions,
            changed,
            music,
        })
    }

    /// Plays or pauses the last music app that was current.
    pub fn pass_tap_on(&self) {
        let last = self.music.lock().expect("the music lock").clone();
        let session = last.and_then(|id| find(&self.sessions, &id));
        match session.map(|s| s.TryTogglePlayPauseAsync()) {
            Some(Ok(_)) => {}
            Some(Err(e)) => tracing::warn!(error = %e, "Cannot pass a headphone tap on"),
            None => tracing::info!("A headphone tap, and no music app to pass it to"),
        }
    }
}

impl Drop for Headphones {
    fn drop(&mut self) {
        let _ = self.controls.RemoveButtonPressed(self.pressed);
        let _ = self.sessions.RemoveCurrentSessionChanged(self.changed);
        let _ = self.controls.SetIsEnabled(false);
    }
}

/// The headphone button a transport control stands for.
fn button(pressed: SystemMediaTransportControlsButton) -> Option<HeadphoneButton> {
    if pressed == SystemMediaTransportControlsButton::Next {
        Some(HeadphoneButton::DoubleTap)
    } else if pressed == SystemMediaTransportControlsButton::Play
        || pressed == SystemMediaTransportControlsButton::Pause
    {
        Some(HeadphoneButton::Tap)
    } else {
        None
    }
}

/// Whether a session is jevons' own: an unpackaged app's is named after its executable.
fn own(id: &HSTRING) -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .is_some_and(|name| name.eq_ignore_ascii_case(&id.to_string_lossy()))
}

/// Remembers the current session, unless it is jevons' own.
fn remember(sessions: &Sessions, music: &Mutex<Option<HSTRING>>) {
    if let Ok(current) = sessions.GetCurrentSession()
        && let Ok(id) = current.SourceAppUserModelId()
        && !own(&id)
    {
        *music.lock().expect("the music lock") = Some(id);
    }
}

fn find(sessions: &Sessions, id: &HSTRING) -> Option<MediaSession> {
    let all = sessions.GetSessions().ok()?;
    (0..all.Size().ok()?)
        .filter_map(|i| all.GetAt(i).ok())
        .find(|s| s.SourceAppUserModelId().is_ok_and(|other| &other == id))
}

/// The sessions Windows knows, for the log: jevons' own is among them.
fn listed(sessions: &Sessions) -> String {
    let Ok(all) = sessions.GetSessions() else {
        return String::new();
    };
    (0..all.Size().unwrap_or(0))
        .filter_map(|i| all.GetAt(i).ok()?.SourceAppUserModelId().ok())
        .map(|id| id.to_string_lossy())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_double_tap_is_next_and_a_tap_is_play_or_pause() {
        use SystemMediaTransportControlsButton as B;
        assert_eq!(button(B::Next), Some(HeadphoneButton::DoubleTap));
        assert_eq!(button(B::Play), Some(HeadphoneButton::Tap));
        assert_eq!(button(B::Pause), Some(HeadphoneButton::Tap));
        assert_eq!(button(B::Previous), None);
        assert_eq!(button(B::Stop), None);
    }
}
