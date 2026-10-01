//! The automations the desktop can run: the library, the approvals from the settings, and the
//! platform layers runs act through. The tray, the flow tree's `run.toml` nodes and the tool
//! host (`script:<name>` tools) all run automations through it.

use super::engine::{Confirm, Progress};
use super::library::{Automation, Library};
use super::run::{self, AutomationError, ErrorKind, RunEnv, RunTrace};
use crate::config::AutomationSettings;
use crate::flow::investigate::Progress as Told;
use crate::platform::{ContextInspector, UiActor};
use adk_core::{Tool, ToolContext, async_trait};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

/// The name tools use for automations: `script:<name>` in flow files, `script__<name>` for the
/// model.
pub const SERVER: &str = "script";

/// One automation as the tray, the tools and the decision model list it.
#[derive(Clone, Debug, PartialEq)]
pub struct Listed {
    pub name: String,
    pub description: String,
    /// The arguments' JSON Schema.
    pub parameters: Value,
    pub approved: bool,
    pub version: String,
}

pub struct AutomationHost {
    dir: PathBuf,
    library: RwLock<Arc<Library>>,
    settings: RwLock<AutomationSettings>,
    inspector: Arc<dyn ContextInspector>,
    actor: Arc<dyn UiActor>,
    /// Answers scripts' `confirm()`; without it they get no.
    confirm: RwLock<Option<Confirm>>,
    /// The next run through a tool asks before each action (a step-by-step run whose
    /// arguments the user says first).
    step_next: AtomicBool,
}

impl AutomationHost {
    pub fn new(
        dir: &Path,
        settings: AutomationSettings,
        inspector: Arc<dyn ContextInspector>,
        actor: Arc<dyn UiActor>,
    ) -> Self {
        Self {
            dir: dir.to_path_buf(),
            library: RwLock::new(Arc::new(Library::load(dir))),
            settings: RwLock::new(settings),
            inspector,
            actor,
            confirm: RwLock::new(None),
            step_next: AtomicBool::new(false),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Reads the library folder again.
    pub fn reload(&self) {
        *self.library.write().expect("the library lock") = Arc::new(Library::load(&self.dir));
    }

    pub fn set_settings(&self, settings: AutomationSettings) {
        *self.settings.write().expect("the settings lock") = settings;
    }

    /// Makes the next run through a tool ask before each action.
    pub fn step_next_run(&self, on: bool) {
        self.step_next
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_confirm(&self, confirm: Option<Confirm>) {
        *self.confirm.write().expect("the confirm lock") = confirm;
    }

    pub fn library(&self) -> Arc<Library> {
        self.library.read().expect("the library lock").clone()
    }

    pub fn settings(&self) -> AutomationSettings {
        self.settings.read().expect("the settings lock").clone()
    }

    pub fn is_approved(&self, automation: &Automation) -> bool {
        automation.is_approved(&self.settings.read().expect("the settings lock").approved)
    }

    /// Every automation, approved or not.
    pub fn list(&self) -> Vec<Listed> {
        let library = self.library();
        library
            .automations
            .values()
            .map(|a| Listed {
                name: a.name.clone(),
                description: a.manifest.description.clone(),
                parameters: a.manifest.parameters(),
                approved: self.is_approved(a),
                version: a.version.clone(),
            })
            .collect()
    }

    /// Whether automation `name` asks before it runs (it does, unless the settings say not).
    pub fn asks(&self, name: &str) -> bool {
        !self
            .settings
            .read()
            .expect("the settings lock")
            .unconfirmed
            .iter()
            .any(|u| u == name)
    }

    /// The flow-node globs that may call `name`; empty allows all.
    pub fn allow(&self, name: &str) -> Vec<String> {
        self.settings
            .read()
            .expect("the settings lock")
            .allow
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Runs an approved automation. It blocks: call it off the async workers.
    pub fn run(
        &self,
        name: &str,
        arguments: &Map<String, Value>,
        progress: Option<Progress>,
        cancel: Arc<AtomicBool>,
        step_by_step: bool,
    ) -> RunTrace {
        let library = self.library();
        let Some(automation) = library.get(name) else {
            return failed(
                name,
                ErrorKind::Invalid,
                format!("there is no automation {name:?}"),
            );
        };
        if !self.is_approved(automation) {
            return failed(
                name,
                ErrorKind::NotApproved,
                format!(
                    "{name} needs approval: approve this version from the tray's Automations \
                     menu first"
                ),
            );
        }
        let env = RunEnv {
            inspector: self.inspector.clone(),
            actor: self.actor.clone(),
            confirm: self.confirm.read().expect("the confirm lock").clone(),
            progress,
            cancel,
            step_by_step,
        };
        let trace = run::run(automation, arguments, env);
        if let Some(error) = &trace.error
            && !matches!(error.kind, ErrorKind::Invalid | ErrorKind::Cancelled)
        {
            self.keep_failure(automation, &trace);
        }
        trace
    }

    /// Saves a failed run's trace with its applications' interface as they are now, to fix the
    /// script against (`failures/<time>.json`, the newest ten kept). It stays on this machine.
    fn keep_failure(&self, automation: &Automation, trace: &RunTrace) {
        let windows: Vec<_> = match self.inspector.windows() {
            Ok(all) => {
                let hands = super::hands::Hands::new(
                    self.inspector.clone(),
                    self.actor.clone(),
                    &automation.manifest.apps,
                );
                all.into_iter()
                    .filter(|w| hands.as_ref().is_ok_and(|h| h.allows(&w.app)))
                    .collect()
            }
            Err(_) => Vec::new(),
        };
        let tree = crate::recorded::RecordedTree::record(&*self.inspector, &windows, 40, 5000)
            .unwrap_or_default();
        let dir = automation.dir.join("failures");
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let text =
            serde_json::to_vec_pretty(&json!({"trace": trace, "tree": tree})).unwrap_or_default();
        let _ = std::fs::write(dir.join(format!("{millis}.json")), text);
        let mut kept: Vec<_> = std::fs::read_dir(&dir)
            .map(|d| d.filter_map(Result::ok).map(|e| e.path()).collect())
            .unwrap_or_default();
        kept.sort();
        let excess = kept.len().saturating_sub(10);
        for old in kept.into_iter().take(excess) {
            let _ = std::fs::remove_file(old);
        }
    }

    /// `script:<name>` tools for the named automations (`*` for all).
    pub fn tools(self: &Arc<Self>, wanted: &str) -> Vec<Arc<dyn Tool>> {
        self.library()
            .automations
            .values()
            .filter(|a| wanted == "*" || a.name == wanted)
            .map(|a| {
                Arc::new(ScriptTool {
                    host: self.clone(),
                    automation: a.name.clone(),
                    name: format!("{SERVER}__{}", a.name),
                    description: a.manifest.description.clone(),
                    parameters: a.manifest.parameters(),
                }) as Arc<dyn Tool>
            })
            .collect()
    }
}

fn failed(name: &str, kind: ErrorKind, message: String) -> RunTrace {
    RunTrace {
        automation: name.to_string(),
        version: String::new(),
        arguments: Value::Null,
        dry_run: false,
        steps: Vec::new(),
        log: Vec::new(),
        actions: Vec::new(),
        result: None,
        error: Some(AutomationError::new(kind, message)),
        replayed: None,
        ms: 0,
    }
}

/// An automation as a tool.
struct ScriptTool {
    host: Arc<AutomationHost>,
    automation: String,
    /// `script__<automation>`, as the model names it.
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl Tool for ScriptTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(self.parameters.clone())
    }

    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        let host = self.host.clone();
        let name = self.automation.clone();
        let arguments = args.as_object().cloned().unwrap_or_default();
        let trace = tokio::task::spawn_blocking(move || {
            let step_by_step = host
                .step_next
                .swap(false, std::sync::atomic::Ordering::Relaxed);
            host.run(
                &name,
                &arguments,
                None,
                Arc::new(AtomicBool::new(false)),
                step_by_step,
            )
        })
        .await
        .map_err(|e| adk_core::AdkError::tool(e.to_string()))?;
        match &trace.error {
            // The answer alone: it is what the bubble or the next node reads.
            None => Ok(match trace.result.clone() {
                Some(Value::Null) | None => json!({"done": true}),
                Some(result) => result,
            }),
            Some(error) => Err(adk_core::AdkError::tool(match (error.line, error.column) {
                (Some(line), Some(column)) => {
                    format!("{} (script.rhai:{line}:{column})", error.message)
                }
                _ => error.message.clone(),
            })),
        }
    }
}

/// Turns a flow step's progress callback into an automation's.
pub fn progress_of(told: Option<Told>) -> Option<Progress> {
    told.map(|told| Arc::new(move |label: &str| told(label)) as Progress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorded::{ReplayActor, tests::slack_demonstration};

    fn library_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-host-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("post")).unwrap();
        std::fs::write(
            dir.join("post/automation.toml"),
            "description = \"Posts a message\"\napps = [\"slack.exe\"]\n[args.channel]\ndescription = \"The channel\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("post/script.rhai"),
            "find(\"//TreeItem[.//Text[@name = $channel]]\").invoke();\n#{ opened: args.channel }\n",
        )
        .unwrap();
        dir
    }

    #[tokio::test]
    async fn only_approved_versions_run_and_tools_carry_their_arguments() {
        let dir = library_dir("approve");
        let (demonstration, _, _) = slack_demonstration();
        let replay = Arc::new(ReplayActor::new(demonstration));
        let host = Arc::new(AutomationHost::new(
            &dir,
            AutomationSettings::default(),
            replay.clone(),
            replay.clone(),
        ));
        let listed = host.list();
        assert_eq!(listed.len(), 1);
        assert!(!listed[0].approved);
        assert_eq!(listed[0].parameters["required"], json!(["channel"]));
        let args = json!({"channel": "random"}).as_object().unwrap().clone();
        let refused = host.run("post", &args, None, Arc::new(AtomicBool::new(false)), false);
        assert_eq!(refused.error.unwrap().kind, ErrorKind::NotApproved);
        assert_eq!(replay.done(), 0);
        let mut settings = AutomationSettings::default();
        settings
            .approved
            .insert("post".into(), listed[0].version.clone());
        host.set_settings(settings);
        let tools = host.tools("*");
        assert_eq!(tools[0].name(), "script__post");
        let ctx: Arc<dyn ToolContext> = Arc::new(adk_tool::SimpleToolContext::new("test"));
        let result = tools[0]
            .execute(ctx, json!({"channel": "random"}))
            .await
            .unwrap();
        assert_eq!(result, json!({"opened": "random"}));
        assert_eq!(replay.done(), 1);
        // Editing the script unpins it.
        std::fs::write(dir.join("post/script.rhai"), "#{ opened: \"x\" }\n").unwrap();
        host.reload();
        assert!(!host.list()[0].approved);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
