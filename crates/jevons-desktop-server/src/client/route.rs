//! Routes: each capability's provider and model, and what the decision provider takes.
//!
//! Flows and machines ask a route, never a provider. A decision goes without what its provider
//! does not take, and in as many requests as its question limit needs; the notes say what was
//! left out, so the take's trace does too.

use super::{Client, ClientError, DecisionRequest, DecisionResponse};
use crate::config::{Extension, Provider};
use serde_json::Value;

/// The probability from which a decision model's choice counts as sure, unless its provider
/// says otherwise. It was tuned on DiffusionGemma through our System One; other models keep it
/// until they are measured.
pub const MIN_PROBABILITY: f64 = 0.7;

/// The most questions asked of a provider that is not ours in one request, unless it says
/// otherwise. OpenRouter documents no limit, so this is a cautious one.
pub const EXTERNAL_MAX_QUESTIONS: usize = 8;

/// What a decision provider takes beyond TypeSafe's contract, and how sure its model is.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// The extensions of ours it takes (`docs/api.md`); the rest are left out of its requests.
    pub steps: bool,
    pub samples: bool,
    pub think: bool,
    /// The most questions one request may ask; ours splits them itself and has no limit.
    pub max_questions: Option<usize>,
    /// A noul question's criteria need both `true` and `false`: the one a question leaves out
    /// goes as `null`. Ours takes either alone, and reads `null` as a description.
    pub both_noul_criteria: bool,
    /// The probability from which its model's choice counts as sure.
    pub min_probability: f64,
}

impl Profile {
    /// Our own System One, embedded or on a jevons server.
    pub fn ours() -> Self {
        Self {
            steps: true,
            samples: true,
            think: true,
            max_questions: None,
            both_noul_criteria: false,
            min_probability: MIN_PROBABILITY,
        }
    }

    /// A System One that is not ours (Jev on OpenRouter): TypeSafe's contract alone.
    pub fn external() -> Self {
        Self {
            steps: false,
            samples: false,
            think: false,
            max_questions: Some(EXTERNAL_MAX_QUESTIONS),
            both_noul_criteria: true,
            min_probability: MIN_PROBABILITY,
        }
    }

    /// The profile of `provider`'s kind, with what the provider sets in its place.
    pub fn of(provider: &Provider) -> Self {
        let mut profile = if provider.kind.names_its_models() {
            Self::ours()
        } else {
            Self::external()
        };
        if let Some(extensions) = &provider.extensions {
            profile.steps = extensions.contains(&Extension::Steps);
            profile.samples = extensions.contains(&Extension::Samples);
            profile.think = extensions.contains(&Extension::Think);
        }
        if let Some(most) = provider.max_questions {
            profile.max_questions = Some(most);
        }
        if let Some(probability) = provider.min_probability {
            profile.min_probability = probability;
        }
        profile
    }
}

/// A capability's provider and the model to ask it for.
#[derive(Clone, Debug)]
pub struct Route {
    pub client: Client,
    pub model: String,
    /// The provider's name in the settings.
    pub provider: String,
    pub profile: Profile,
}

impl Route {
    /// A route to our own API at `client`, as the embedded provider and a jevons server are.
    pub fn ours(client: Client, provider: &str, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
            provider: provider.into(),
            profile: Profile::ours(),
        }
    }

    /// What traces call it, such as `openrouter/typesafe/jev-1.13`.
    pub fn name(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }

    /// `request` as this provider takes it, and a note for each extension left out and for a
    /// request that goes in several.
    pub fn fit(&self, mut request: DecisionRequest) -> (DecisionRequest, Vec<String>) {
        let name = self.name();
        let mut notes = Vec::new();
        let mut drop = |taken: bool, set: &mut Option<u32>, extension: &str| {
            if !taken && set.take().is_some() {
                notes.push(format!("{extension} dropped: {name} does not support it"));
            }
        };
        drop(self.profile.steps, &mut request.steps, "steps");
        drop(self.profile.samples, &mut request.samples, "samples");
        drop(self.profile.think, &mut request.think, "think");
        let asked = request.questions.len();
        if let Some(most) = self.profile.max_questions
            && asked > most
        {
            notes.push(format!(
                "{asked} questions asked in {} requests: {name} takes {most} in one",
                asked.div_ceil(most)
            ));
        }
        (request, notes)
    }

    /// Asks a fitted request's questions, in as many requests as the provider's limit needs,
    /// one after the other; the answers come back as one response.
    pub async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, ClientError> {
        let keys: Vec<&String> = request.questions.keys().collect();
        let most = self.profile.max_questions.unwrap_or(usize::MAX).max(1);
        let mut whole: Option<DecisionResponse> = None;
        for keys in keys.chunks(most) {
            let part = DecisionRequest {
                model: request.model.clone(),
                state: request.state.clone(),
                questions: keys
                    .iter()
                    .map(|key| ((*key).clone(), request.questions[*key].clone()))
                    .collect(),
                steps: request.steps,
                samples: request.samples,
                think: request.think,
            };
            let mut body = serde_json::to_value(&part)
                .map_err(|e| ClientError::Protocol(format!("System One request: {e}")))?;
            if self.profile.both_noul_criteria {
                both_noul_criteria(&mut body);
            }
            let response = self.client.decide_body(&body).await?;
            whole = Some(match whole {
                None => response,
                Some(mut whole) => {
                    whole.answers.extend(response.answers);
                    whole.usage.input_tokens += response.usage.input_tokens;
                    whole.usage.output_tokens += response.usage.output_tokens;
                    whole
                }
            });
        }
        whole.ok_or_else(|| ClientError::Protocol("a decision needs a question".into()))
    }
}

/// Gives each noul question's criteria both keys, `null` for the one it lacks.
fn both_noul_criteria(body: &mut Value) {
    let Some(questions) = body["questions"].as_object_mut() else {
        return;
    };
    for question in questions.values_mut() {
        if question["type"] != "noul" {
            continue;
        }
        if let Some(criteria) = question.get_mut("criteria").and_then(Value::as_object_mut) {
            for key in ["true", "false"] {
                criteria.entry(key).or_insert(Value::Null);
            }
        }
    }
}

/// Each capability's route; `None` where nothing serves it.
#[derive(Clone, Debug, Default)]
pub struct Routes {
    /// Uploaded audio to text.
    pub speech: Option<Route>,
    /// Streamed audio to text, tried first; uploads are the fallback.
    pub realtime: Option<Route>,
    pub decision: Option<Route>,
    pub generation: Option<Route>,
}

#[cfg(test)]
mod tests {
    use super::super::{Answer, NoulCriteria, Question, probability_of};
    use super::*;
    use crate::config::ProviderKind;
    use axum::{Json, Router, routing::post};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    /// A System One that answers yes to everything and keeps the bodies it got. With
    /// `strict`, it takes TypeSafe's contract alone, as OpenRouter documents it: an extension
    /// or a noul criteria with one key is a 400.
    async fn provider(strict: bool) -> (Client, Arc<Mutex<Vec<Value>>>) {
        let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
        let kept = seen.clone();
        let app = Router::new().route(
            "/v1/systemone",
            post(move |Json(body): Json<Value>| {
                let kept = kept.clone();
                async move {
                    kept.lock().unwrap().push(body.clone());
                    let questions = body["questions"].as_object().unwrap();
                    let extended = ["steps", "samples", "think"]
                        .iter()
                        .any(|field| body.get(*field).is_some());
                    let half = questions.values().any(|q| {
                        q["type"] == "noul"
                            && q["criteria"].as_object().is_some_and(|c| c.len() != 2)
                    });
                    if strict && (extended || half) {
                        let error = json!({"error": {"message": "Invalid request"}});
                        return (axum::http::StatusCode::BAD_REQUEST, Json(error));
                    }
                    let answers: serde_json::Map<String, Value> = questions
                        .keys()
                        .map(|key| (key.clone(), json!({"type": "noul", "noul": 0.9})))
                        .collect();
                    let body = json!({
                        "id": "gen-1", "provider": "TypeSafe", "model": "typesafe/jev-1.13",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.00002},
                        "answers": answers,
                    });
                    (axum::http::StatusCode::OK, Json(body))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (Client::new(&base, None), seen)
    }

    fn noul(text: &str) -> Question {
        Question::Noul {
            instructions: Some(text.into()),
            criteria: Some(NoulCriteria {
                yes: Some(text.into()),
                no: None,
            }),
        }
    }

    fn request(questions: usize) -> DecisionRequest {
        DecisionRequest {
            model: "typesafe/jev-1.13".into(),
            state: "what was said".into(),
            questions: (0..questions)
                .map(|i| (format!("q{i:02}"), noul("Is it so?")))
                .collect(),
            steps: Some(4),
            samples: None,
            think: Some(64),
        }
    }

    fn jev(client: Client) -> Route {
        Route {
            client,
            model: "typesafe/jev-1.13".into(),
            provider: "openrouter".into(),
            profile: Profile::of(&Provider::of(ProviderKind::Openrouter)),
        }
    }

    #[test]
    fn a_provider_s_kind_gives_its_profile_and_its_settings_replace_parts() {
        for kind in [ProviderKind::Embedded, ProviderKind::Jevons] {
            assert_eq!(Profile::of(&Provider::of(kind)), Profile::ours());
        }
        for kind in [ProviderKind::Openrouter, ProviderKind::OpenaiCompatible] {
            assert_eq!(Profile::of(&Provider::of(kind)), Profile::external());
        }
        let ours = Profile::ours();
        assert!(ours.steps && ours.samples && ours.think);
        assert_eq!((ours.max_questions, ours.min_probability), (None, 0.7));
        let external = Profile::external();
        assert!(!external.steps && !external.samples && !external.think);
        assert_eq!(external.max_questions, Some(8));
        assert!(external.both_noul_criteria && !ours.both_noul_criteria);
        // What a provider sets replaces that part of its kind's profile.
        let set = Provider {
            extensions: Some(vec![Extension::Think]),
            max_questions: Some(3),
            min_probability: Some(0.55),
            ..Provider::of(ProviderKind::Openrouter)
        };
        let profile = Profile::of(&set);
        assert!(profile.think && !profile.steps && !profile.samples);
        assert_eq!(
            (profile.max_questions, profile.min_probability),
            (Some(3), 0.55)
        );
        assert!(profile.both_noul_criteria);
    }

    #[tokio::test]
    async fn extensions_a_provider_lacks_are_dropped_with_a_note() {
        let (client, seen) = provider(true).await;
        let route = jev(client);
        let (fitted, notes) = route.fit(request(2));
        assert_eq!(
            notes,
            [
                "steps dropped: openrouter/typesafe/jev-1.13 does not support it",
                "think dropped: openrouter/typesafe/jev-1.13 does not support it",
            ]
        );
        assert_eq!(
            (fitted.steps, fitted.samples, fitted.think),
            (None, None, None)
        );
        let response = route.decide(&fitted).await.unwrap();
        assert_eq!(response.answers.len(), 2);
        assert_eq!(response.answers["q00"], Answer::Noul { noul: 0.9 });
        // What went has no extension, and both noul criteria.
        let bodies = seen.lock().unwrap().clone();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].get("steps").is_none() && bodies[0].get("think").is_none());
        assert_eq!(
            bodies[0]["questions"]["q00"]["criteria"],
            json!({"true": "Is it so?", "false": null})
        );
        // Our own takes them all, and one noul criterion alone: nothing to note or change.
        let (client, seen) = provider(false).await;
        let ours = Route::ours(client, "embedded", "jev");
        let (fitted, notes) = ours.fit(request(2));
        assert_eq!(notes, Vec::<String>::new());
        assert_eq!((fitted.steps, fitted.think), (Some(4), Some(64)));
        ours.decide(&fitted).await.unwrap();
        let bodies = seen.lock().unwrap().clone();
        assert_eq!(
            (bodies[0]["steps"].clone(), bodies[0]["think"].clone()),
            (json!(4), json!(64))
        );
        assert_eq!(
            bodies[0]["questions"]["q00"]["criteria"],
            json!({"true": "Is it so?"})
        );
        // Unfitted, the strict provider refuses the request.
        let refused = route.decide(&request(1)).await;
        assert!(matches!(refused, Err(ClientError::Api { status: 400, .. })));
    }

    #[tokio::test]
    async fn questions_over_a_provider_s_limit_go_in_several_requests() {
        let (client, seen) = provider(true).await;
        let mut route = jev(client);
        route.profile.max_questions = Some(5);
        let (fitted, notes) = route.fit(DecisionRequest {
            steps: None,
            think: None,
            ..request(12)
        });
        assert_eq!(
            notes,
            ["12 questions asked in 3 requests: openrouter/typesafe/jev-1.13 takes 5 in one"]
        );
        let response = route.decide(&fitted).await.unwrap();
        // Every question is answered once, and the usage is all the requests'.
        let keys: Vec<String> = (0..12).map(|i| format!("q{i:02}")).collect();
        assert_eq!(response.answers.keys().cloned().collect::<Vec<_>>(), keys);
        assert_eq!(
            (response.usage.input_tokens, response.usage.output_tokens),
            (30, 6)
        );
        let sizes: Vec<usize> = seen
            .lock()
            .unwrap()
            .iter()
            .map(|body| body["questions"].as_object().unwrap().len())
            .collect();
        assert_eq!(sizes, [5, 5, 2]);
        // At the limit it is one request, with no note.
        let (_, notes) = route.fit(DecisionRequest {
            steps: None,
            think: None,
            ..request(5)
        });
        assert_eq!(notes, Vec::<String>::new());
    }

    #[test]
    fn an_external_answer_parses_with_what_it_leaves_out() {
        // OpenRouter's response carries an id, a provider and a cost, and may leave out a
        // choice's probabilities and confidence.
        let body = json!({
            "id": "gen-dec-1", "provider": "TypeSafe", "model": "typesafe/jev-1.13-20260917",
            "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
            "answers": {
                "team": {"type": "choice", "choice": "payments"},
                "sure": {"type": "choice", "choice": "a", "confidence": 0.75,
                         "probabilities": {"a": 0.84, "b": 0.16}},
                "urgency": {"type": "score", "score": 1.99},
            },
        });
        let response: DecisionResponse = serde_json::from_value(body).unwrap();
        let Answer::Choice {
            choice,
            probabilities,
            confidence,
        } = &response.answers["team"]
        else {
            panic!("a choice");
        };
        assert_eq!((choice.as_str(), confidence), ("payments", &None));
        assert_eq!(probabilities, &BTreeMap::new());
        // The probability of a choice: its own, else the confidence, else taken as given.
        assert_eq!(probability_of(probabilities, choice, *confidence), 1.0);
        assert_eq!(probability_of(probabilities, choice, Some(0.75)), 0.75);
        let own = BTreeMap::from([("payments".to_string(), 0.84)]);
        assert_eq!(probability_of(&own, choice, Some(0.75)), 0.84);
        assert!(matches!(response.answers["sure"], Answer::Choice { .. }));
        assert!(matches!(response.answers["urgency"], Answer::Score { .. }));
    }
}
