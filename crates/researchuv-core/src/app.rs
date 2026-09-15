//! Application state and task execution.
//!
//! [`App`] dispatches registered operations through their begin, task, and end
//! hooks. It holds the current document and configuration, collects task reports,
//! and records island snapshots for undo and redo.

use crate::model::{Island, MultiMesh};
use crate::val::{CRef, Val};
use std::collections::BTreeMap;

/// Coordinate-space profile carried by the application state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edition {
    RealSpace,
    VirtualSpace,
}

impl Edition {
    /// Short label for the selected coordinate-space profile.
    pub fn as_str(self) -> &'static str {
        match self {
            Edition::RealSpace => "RS",
            Edition::VirtualSpace => "VS",
        }
    }
}

/// Configuration store with dot-separated keys and typed [`Val`] values.
#[derive(Clone, Debug, Default)]
pub struct Config {
    values: BTreeMap<String, Val>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, v: Val) -> &mut Self {
        self.values.insert(key.to_string(), v);
        self
    }

    pub fn get(&self, key: &str) -> Option<&Val> {
        self.values.get(key)
    }

    /// All known keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.values.keys()
    }

    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.values.get(key) {
            Some(Val::Bool(b)) => *b,
            Some(Val::Int(i)) => *i != 0,
            Some(Val::Double(d)) => *d != 0.0,
            _ => default,
        }
    }

    pub fn get_i64(&self, key: &str, default: i64) -> i64 {
        match self.values.get(key) {
            Some(Val::Int(i)) => *i,
            Some(Val::Double(d)) => *d as i64,
            _ => default,
        }
    }

    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        match self.values.get(key) {
            Some(Val::Double(d)) => *d,
            Some(Val::Int(i)) => *i as f64,
            _ => default,
        }
    }

    /// Seed the supported pipeline settings with their default values.
    pub fn seed_defaults(&mut self) {
        // LSCM conformal/area mix: A = #AngleDistanceMix (default 1.0 ⇒ pure angle
        // distance), B = 1 - A. (ALGORITHMS.md §3, `CIsomap::ComputeLs`.)
        self.set("Prefs.Optimize.AngleDistanceMix", Val::Double(1.0));
        self.set("Prefs.Optimize.Mix", Val::Double(0.5));
        // Packing backend preference; the current implementation uses the CPU.
        self.set("Prefs.PackOptions.UseGPU", Val::Bool(false));
        self.set("Prefs.PackOptions.MixScales", Val::Bool(false));
        // Seam selection: dihedral angle threshold in degrees
        // (`CTaskCut` cuts edges whose dihedral exceeds `Auto.SharpEdges.AngleMin`).
        self.set("Vars.AutoSelect.SharpEdges.Angle", Val::Double(30.0));
        self.set("Vars.AutoSelect.SharpEdges.UseGeoNormals", Val::Bool(false));
        // Import behaviour.
        self.set("Prefs.File.ImportAutoWeld", Val::Bool(true));
    }
}

/// Results and diagnostics returned by a dispatched task.
#[derive(Clone, Debug, Default)]
pub struct DataReport {
    /// `#Result` — `0` = success, nonzero = error code.
    pub result: i64,
    /// Human-readable messages/warnings appended by the task.
    pub messages: Vec<String>,
    /// Free-form scalar results (e.g. iteration counts, areas).
    pub extras: Vec<(String, Val)>,
}

impl DataReport {
    pub fn note(&mut self, msg: impl Into<String>) {
        self.messages.push(msg.into());
    }

    pub fn extra(&mut self, key: &str, v: Val) {
        self.extras.push((key.to_string(), v));
    }
}

/// `CTask` — base trait for every engine operation.
///
/// [`App::do_task`] invokes `begin` → `task` → `end`, then (if requested
/// and the task is undoable) pushes an undo record. The task is looked up by name in the
/// registry before dispatch.
pub trait Task {
    /// Task name as used in `CRef` dispatch (e.g. `"Unfold"`, `"Weld"`, `"Cut"`, `"Pack"`).
    fn name(&self) -> &str;

    /// `CTask::Begin` — validate parameters, prepare state.
    fn begin(
        &mut self,
        _app: &mut App,
        _params: &CRef,
        _report: &mut DataReport,
    ) -> Result<(), String> {
        Ok(())
    }

    /// `CTask::Task` — perform the work.
    fn task(
        &mut self,
        app: &mut App,
        params: &CRef,
        report: &mut DataReport,
    ) -> Result<(), String>;

    /// `CTask::End` — cleanup / finalization.
    fn end(
        &mut self,
        _app: &mut App,
        _params: &CRef,
        _report: &mut DataReport,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Whether this task records an undo entry (CTask undo counter at `+0x360`).
    fn undoable(&self) -> bool {
        true
    }
}

/// One engine instance: coordinate-space profile, config, document, task registry,
/// and the undo/redo stacks.
pub struct App {
    pub edition: Edition,
    pub config: Config,
    /// The current multi-mesh document (None until a mesh is imported).
    pub document: Option<MultiMesh>,
    registry: BTreeMap<String, Box<dyn Task>>,
    undo: Vec<(String, Vec<Island>)>,
    redo: Vec<(String, Vec<Island>)>,
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("edition", &self.edition)
            .field("document", &self.document.is_some())
            .field("tasks", &self.task_names())
            .field("undo", &self.undo.len())
            .field("redo", &self.redo.len())
            .finish()
    }
}

impl App {
    pub fn new(edition: Edition) -> Self {
        let mut config = Config::new();
        config.seed_defaults();
        Self {
            edition,
            config,
            document: None,
            registry: BTreeMap::new(),
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Register a task under its `Task::name()`.
    pub fn register<T: Task + 'static>(&mut self, task: T) {
        let name = task.name().to_string();
        self.registry.insert(name, Box::new(task));
    }

    /// Names of all registered tasks (sorted).
    pub fn task_names(&self) -> Vec<&str> {
        self.registry.keys().map(|s| s.as_str()).collect()
    }

    fn islands_snapshot(&self) -> Vec<Island> {
        self.document.as_ref().map(|d| d.islands.clone()).unwrap_or_default()
    }

    /// Restore the current islands from a snapshot.
    fn restore_islands(&mut self, islands: Vec<Island>) {
        match &mut self.document {
            Some(d) => d.islands = islands,
            None => {}
        }
    }

    /// Dispatch a named task with typed parameters.
    ///
    /// Resolves `name` in the registry, drives `Begin → Task → End`, and — when
    /// `with_undo` and the task is undoable — pushes an undo/redo pair so the result is
    /// reversible.
    pub fn do_task(&mut self, name: &str, params: &CRef, with_undo: bool) -> Result<DataReport, String> {
        let mut report = DataReport::default();
        let task_name = name.to_string();
        // Temporarily remove the task so `self` can be borrowed mutably inside the closures.
        let mut task = match self.registry.remove(&task_name) {
            Some(t) => t,
            None => return Err(format!("unknown task '{name}'")),
        };

        let run = (|| -> Result<(DataReport, Option<Vec<Island>>), String> {
            task.begin(self, params, &mut report)?;
            let snapshot = if with_undo && task.undoable() {
                Some(self.islands_snapshot())
            } else {
                None
            };
            let r = task
                .task(self, params, &mut report)
                .and_then(|()| task.end(self, params, &mut report));
            r.map(|()| (report, snapshot))
        })();

        self.registry.insert(task_name, task);
        let (mut report, snapshot) = run?;

        if let Some(before) = snapshot {
            let after = self.islands_snapshot();
            self.undo.push((name.to_string(), before));
            self.redo.push((name.to_string(), after));
        }
        report.result = 0;
        Ok(report)
    }

    /// `CTask::Undo` — restore the previous island state; returns the undone task name.
    pub fn undo(&mut self) -> Option<String> {
        if let Some((label, before)) = self.undo.pop() {
            let after = self.islands_snapshot();
            self.restore_islands(before);
            self.redo.push((label.clone(), after));
            Some(label)
        } else {
            None
        }
    }

    /// `CTask::Redo` — re-apply a previously undone task; returns the redone task name.
    pub fn redo(&mut self) -> Option<String> {
        if let Some((label, after)) = self.redo.pop() {
            let before = self.islands_snapshot();
            self.restore_islands(after);
            self.undo.push((label.clone(), before));
            Some(label)
        } else {
            None
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SurfaceMesh;
    use researchuv_math::Vec2;

    /// A trivial undoable task that records a tag and mutates the document's island UVs.
    struct TagTask {
        tag: &'static str,
    }

    impl Task for TagTask {
        fn name(&self) -> &str {
            self.tag
        }

        fn task(&mut self, app: &mut App, _params: &CRef, report: &mut DataReport) -> Result<(), String> {
            let doc = app
                .document
                .as_mut()
                .ok_or_else(|| "TagTask requires an open document".to_string())?;
            for isl in doc.islands.iter_mut() {
                for uv in isl.uv.iter_mut() {
                    uv.v += 1.0;
                }
            }
            report.note(format!("TagTask ran: {}", self.tag));
            Ok(())
        }
    }

    /// A task that fails in `End` — must still restore registry + report the error.
    struct FailingEnd;

    impl Task for FailingEnd {
        fn name(&self) -> &str {
            "FailingEnd"
        }

        fn task(&mut self, _app: &mut App, _params: &CRef, _report: &mut DataReport) -> Result<(), String> {
            Ok(())
        }

        fn end(&mut self, _app: &mut App, _params: &CRef, _report: &mut DataReport) -> Result<(), String> {
            Err("end failed".into())
        }
    }

    fn app_with_island() -> App {
        let mut app = App::new(Edition::RealSpace);
        // Tiny single-triangle "mesh" (not a valid half-edge manifold — fine for this test).
        let src = SurfaceMesh {
            positions: vec![
                researchuv_math::Vec3::new(0.0, 0.0, 0.0),
                researchuv_math::Vec3::new(1.0, 0.0, 0.0),
                researchuv_math::Vec3::new(0.0, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2]],
            halfedges: Vec::new(),
        };
        let mut doc = MultiMesh::new(src);
        doc.islands.push(Island {
            positions: vec![
                researchuv_math::Vec3::new(0.0, 0.0, 0.0),
                researchuv_math::Vec3::new(1.0, 0.0, 0.0),
                researchuv_math::Vec3::new(0.0, 1.0, 0.0),
            ],
            uv: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)],
            tris: vec![[0, 1, 2]],
            source_vertex_ids: vec![0, 1, 2],
            border: vec![],
        });
        app.document = Some(doc);
        app
    }

    #[test]
    fn config_defaults_are_seeded() {
        let app = App::new(Edition::RealSpace);
        assert_eq!(app.edition.as_str(), "RS");
        assert!(app.config.get_bool("Prefs.File.ImportAutoWeld", false));
        assert!(!app.config.get_bool("Prefs.PackOptions.UseGPU", false));
        assert!((app.config.get_f64("Prefs.Optimize.AngleDistanceMix", -1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unknown_task_is_rejected() {
        let mut app = App::new(Edition::VirtualSpace);
        let params = CRef::new("Params");
        assert!(app.do_task("Nope", &params, false).is_err());
        // Registry untouched.
        assert!(app.task_names().is_empty());
    }

    #[test]
    fn do_task_runs_and_undoes() {
        let mut app = app_with_island();
        app.register(TagTask { tag: "ShiftUV" });
        assert_eq!(app.task_names(), vec!["ShiftUV"]);

        let params = CRef::new("Params");
        let report = app.do_task("ShiftUV", &params, true).expect("task should run");
        assert_eq!(report.result, 0);
        assert!(report.messages.iter().any(|m| m.contains("ShiftUV")));
        assert!(app.can_undo());

        // uv.v should now be shifted by 1.
        let doc = app.document.as_ref().unwrap();
        assert!((doc.islands[0].uv[1].v - 1.0).abs() < 1e-12);

        let undone = app.undo();
        assert_eq!(undone.as_deref(), Some("ShiftUV"));
        let doc = app.document.as_ref().unwrap();
        assert!(doc.islands[0].uv[1].v.abs() < 1e-12, "undo restored UV");
        assert!(app.can_redo());

        let redone = app.redo();
        assert_eq!(redone.as_deref(), Some("ShiftUV"));
        let doc = app.document.as_ref().unwrap();
        assert!((doc.islands[0].uv[1].v - 1.0).abs() < 1e-12, "redo re-applied UV");
    }

    #[test]
    fn failing_end_reports_error_and_keeps_registry() {
        let mut app = app_with_island();
        app.register(FailingEnd);
        let params = CRef::new("Params");
        let err = app.do_task("FailingEnd", &params, true).unwrap_err();
        assert_eq!(err, "end failed");
        // Task remains registered and dispatchable again.
        assert_eq!(app.task_names(), vec!["FailingEnd"]);
    }
}
