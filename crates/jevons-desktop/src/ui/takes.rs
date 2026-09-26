//! The recent takes, step by step.

use crate::agent::View;
use eframe::egui;

pub fn show(ui: &mut egui::Ui, view: &View) {
    if view.traces.is_empty() {
        ui.label(format!(
            "No takes yet. Tap {} to toggle dictation, or hold it while speaking.",
            view.config.dictation.hotkey
        ));
        return;
    }
    for trace in &view.traces {
        let status = match &trace.error {
            Some(e) => format!("✘ {e}"),
            None => format!("✔ {}", trace.output.chars().take(80).collect::<String>()),
        };
        egui::CollapsingHeader::new(format!(
            "#{} · {} · {:.1} s · {status}",
            trace.take, trace.context.app.process_name, trace.audio_seconds
        ))
        .id_salt(("take", trace.take))
        .show(ui, |ui| {
            egui::Grid::new(("take-grid", trace.take))
                .num_columns(2)
                .show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, key: &str, value: String| {
                        ui.strong(key);
                        ui.label(value);
                        ui.end_row();
                    };
                    row(ui, "Transcript", trace.transcript.clone());
                    if let Some(path) = trace.transcription {
                        row(ui, "Transcribed by", format!("{path:?}"));
                    }
                    if let Some(resolution) = &trace.resolution {
                        row(
                            ui,
                            "Profile",
                            match &resolution.destination {
                                Some(d) => format!("{} → {d}", resolution.profile),
                                None => resolution.profile.clone(),
                            },
                        );
                    }
                    if let Some(action) = trace.action {
                        row(ui, "Action", format!("{action:?}"));
                    }
                    row(ui, "Output", trace.output.clone());
                    if let Some(delivery) = &trace.delivery {
                        row(ui, "Delivery", format!("{delivery:?}"));
                    }
                    let timings: Vec<String> = trace
                        .timings
                        .iter()
                        .map(|(step, ms)| format!("{step} {ms} ms"))
                        .collect();
                    row(ui, "Timings", timings.join(" · "));
                });
            for note in &trace.notes {
                ui.colored_label(ui.visuals().warn_fg_color, note);
            }
            let value = serde_json::to_value(trace).unwrap_or_default();
            egui::CollapsingHeader::new("Full trace (context, decision, prompt)")
                .id_salt(("trace-json", trace.take))
                .show(ui, |ui| super::json_tree(ui, ("trace", trace.take), &value));
            if ui.button("Copy trace as JSON").clicked() {
                ui.ctx()
                    .copy_text(serde_json::to_string_pretty(&value).unwrap_or_default());
            }
        });
    }
}
