//! Files jevons writes into a folder the user edits (the automations library's guides): each
//! is kept up to date until someone edits it, and then left alone. The flows folder has the
//! same rule on the server's side.

use sha2::{Digest, Sha256};
use std::path::Path;

/// What writing a folder's files did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitReport {
    /// Files written, by path under the folder.
    pub written: Vec<String>,
    /// Files removed, by path under the folder.
    pub removed: Vec<String>,
    /// Things the user may want to know, such as an `AGENTS.md` left alone.
    pub notes: Vec<String>,
}

impl InitReport {
    /// Every file written or removed, for committing them.
    pub fn changed(&self) -> impl Iterator<Item = &String> {
        self.written.iter().chain(&self.removed)
    }
}

fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Writes a guide jevons keeps up to date, when it is missing or still exactly what jevons
/// wrote (its first line carries the hash of the rest). One the user edited is left alone.
pub fn write_guarded(
    dir: &Path,
    name: &str,
    header: &str,
    body: &str,
    in_repository: &str,
    report: &mut InitReport,
) -> std::io::Result<()> {
    let file = dir.join(name);
    let fresh = format!(
        "{header}{}; jevons updates this file until you edit it) -->\n{body}",
        digest(body)
    );
    match std::fs::read_to_string(&file) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
        Ok(text) if text == fresh => return Ok(()),
        Ok(text) => {
            let unedited = text
                .strip_prefix(header)
                .and_then(|rest| rest.split_once(';'))
                .zip(text.split_once('\n'))
                .is_some_and(|((hash, _), (_, body))| digest(body) == hash);
            if !unedited {
                report.notes.push(format!(
                    "{name} was edited, so jevons no longer updates it; the current one is in \
                     the jevons repository's {in_repository}"
                ));
                return Ok(());
            }
        }
    }
    std::fs::write(&file, fresh)?;
    report.written.push(name.into());
    Ok(())
}
