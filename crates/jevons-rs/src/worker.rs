//! Bounded inference queue and the dedicated thread that owns the native model.

use crate::error::ApiError;
use axum::http::StatusCode;
use jevons_engine::{Engine, Error, ModelConfig, ModelInfo, ThinkDecoding};
use jevons_system_one::{Request, Response, ValidationError};
use std::thread::JoinHandle;
use tokio::sync::{mpsc, oneshot};

pub(crate) struct Job {
    pub request: Request,
    pub reply: oneshot::Sender<Result<Response, ApiError>>,
}

#[derive(Clone)]
pub struct Client {
    pub(crate) sender: mpsc::Sender<Job>,
}

impl Client {
    pub async fn read(&self, request: Request) -> Result<Response, ApiError> {
        let (reply, receiver) = oneshot::channel();
        self.sender
            .try_send(Job { request, reply })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => ApiError::overloaded(),
                mpsc::error::TrySendError::Closed(_) => ApiError::unavailable(),
            })?;
        receiver.await.map_err(|_| ApiError::unavailable())?
    }

    pub fn is_alive(&self) -> bool {
        !self.sender.is_closed()
    }
}

/// The model is loaded, used, and dropped on this thread; no unsafe Send implementation is needed.
pub async fn start(
    config: ModelConfig,
    model_id: String,
    seed: u64,
    capacity: usize,
    think_decoding: ThinkDecoding,
) -> Result<(Client, JoinHandle<()>, ModelInfo), Box<dyn std::error::Error>> {
    if capacity == 0 {
        return Err("Queue capacity must be positive".into());
    }
    let (sender, mut receiver) = mpsc::channel::<Job>(capacity);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("diffusion-inference".into())
        .spawn(move || {
            let mut engine = match Engine::load(&config).and_then(|mut engine| {
                engine.set_think_decoding(think_decoding)?;
                Ok(engine)
            }) {
                Ok(engine) => engine,
                Err(error) => {
                    let _ = ready_sender.send(Err(error.to_string()));
                    return;
                }
            };
            if ready_sender.send(Ok(engine.model_info().clone())).is_err() {
                return;
            }
            while let Some(job) = receiver.blocking_recv() {
                if job.reply.is_closed() {
                    continue;
                }
                let result = evaluate(&mut engine, &job.request, &model_id, seed);
                let _ = job.reply.send(result);
            }
        })?;
    let info = ready_receiver.await??;
    Ok((Client { sender }, thread, info))
}

fn evaluate(
    engine: &mut Engine,
    request: &Request,
    model_id: &str,
    seed: u64,
) -> Result<Response, ApiError> {
    let input = request.compile(engine.codes()).map_err(|e| {
        ApiError::from(ValidationError::new(
            &["body", "questions"],
            e.to_string(),
            "value_error",
        ))
    })?;
    let read = engine
        .read_with_options(&input, seed, request.options(), request.images())
        .map_err(|error| match error {
            Error::InvalidInput(message) => {
                ApiError::from(ValidationError::new(&["body"], message, "value_error"))
            }
            other => {
                tracing::error!(error = %other, "Inference failed");
                ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "Inference failed",
                )
            }
        })?;
    tracing::info!(
        prompt_tokens = read.prompt_tokens,
        canvas_tokens = read.canvas_tokens,
        forward_ms = read.forward_ms,
        "Restricted canvas read completed"
    );
    request.response(model_id, &read).map_err(|error| {
        tracing::error!(%error, "Answer mapping failed");
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "Answer mapping failed",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use serde_json::json;

    #[tokio::test]
    async fn worker_lost_after_accepting_a_job_returns_unavailable() {
        let (sender, mut receiver) = mpsc::channel(1);
        let client = Client { sender };
        let request =
            Request::parse(json!({"model":"local","state":"x","questions":{"q":{"type":"noul"}}}))
                .unwrap();
        let read = client.read(request);
        let stop_worker = async move {
            let job = receiver.recv().await.unwrap();
            drop(job);
            drop(receiver);
        };
        let (result, ()) = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            tokio::join!(read, stop_worker)
        })
        .await
        .expect("Losing the worker must resolve pending reads");
        assert_eq!(result.unwrap_err().status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(!client.is_alive());
    }

    #[tokio::test]
    async fn queue_saturation_and_worker_failure_are_reported() {
        let (sender, mut receiver) = mpsc::channel(1);
        let client = Client {
            sender: sender.clone(),
        };
        let request = jevons_system_one::Request::parse(
            json!({"model":"local","state":"x","questions":{"q":{"type":"noul"}}}),
        )
        .unwrap();
        let (reply, _reply_receiver) = tokio::sync::oneshot::channel();
        sender
            .try_send(Job {
                request: request.clone(),
                reply,
            })
            .unwrap();
        let error = client.read(request.clone()).await.unwrap_err();
        assert_eq!(error.status.as_u16(), 529);
        assert_eq!(error.into_response().headers()["retry-after"], "1");
        receiver.close();
        assert_eq!(
            client.read(request).await.unwrap_err().status,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
