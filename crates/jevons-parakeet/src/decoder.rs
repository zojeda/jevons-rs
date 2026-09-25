//! The TDT prediction network (an LSTM over emitted tokens), the joint network, and greedy
//! token-and-duration decoding.
//!
//! Decoding is sequential in emitted tokens, so the joint scores a window of frames against the
//! current prediction in one product and the host walks the window: blanks jump ahead by their
//! predicted duration without touching the device, and only an emitted token (which changes
//! the prediction) costs another round trip. Weights stay f32 so greedy choices match the
//! reference.
use crate::config::Config;
use jevons_burn::activation::{relu, sigmoid};
use jevons_burn::layers::{greedy, linear};
use jevons_burn::weights::{Loader, WeightError};
use jevons_burn::{Device, Int, Tensor, TensorData};

/// Frames the joint scores per round trip.
const LOOKAHEAD: usize = 32;

/// One step of greedy decoding: the joint's choice at a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Choice {
    pub token: usize,
    pub logprob: f32,
    pub duration: usize,
}

/// An emitted token at its encoder frame, with the frames it spans (at least one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emission {
    pub token: usize,
    pub frame: usize,
    pub frames: usize,
    pub logprob: f32,
}

/// Walks the frames with the joint's choices: blanks advance by their duration (at least one),
/// tokens are emitted at their frame and advance by theirs (possibly zero, at most
/// `max_symbols` times in a row at one frame). `choose(frame, emitted)` returns the choice at
/// `frame` after the `emitted` tokens.
pub fn walk(
    frames: usize,
    blank: usize,
    max_symbols: usize,
    mut choose: impl FnMut(usize, &[Emission]) -> Choice,
) -> Vec<Emission> {
    let mut emitted = Vec::new();
    let (mut t, mut at_frame) = (0, 0);
    while t < frames {
        let choice = choose(t, &emitted);
        if choice.token == blank {
            t += choice.duration.max(1);
            at_frame = 0;
            continue;
        }
        emitted.push(Emission {
            token: choice.token,
            frame: t,
            frames: choice.duration.max(1),
            logprob: choice.logprob,
        });
        at_frame += 1;
        if choice.duration > 0 || at_frame >= max_symbols {
            t += choice.duration.max(1);
            at_frame = 0;
        }
    }
    emitted
}

struct Lstm {
    input: Tensor<2>,
    hidden: Tensor<2>,
    /// `b_ih + b_hh`, `[1, 4·hidden]`.
    bias: Tensor<2>,
}

pub struct Decoder {
    device: Device,
    hidden: usize,
    vocab: usize,
    blank: usize,
    max_symbols: usize,
    embed: Tensor<2>,
    layers: Vec<Lstm>,
    projector: Tensor<2>,
    projector_bias: Tensor<2>,
    encoder_projector: Tensor<2>,
    encoder_projector_bias: Tensor<2>,
    head: Tensor<2>,
    head_bias: Tensor<2>,
    outputs: usize,
}

/// Per-layer LSTM state `[1, hidden]`, after `tokens` emitted tokens.
struct State {
    h: Vec<Tensor<2>>,
    c: Vec<Tensor<2>>,
    tokens: usize,
}

impl Decoder {
    pub fn load(load: &Loader, config: &Config) -> Result<Self, WeightError> {
        let h = config.decoder_hidden_size;
        let d = config.encoder_config.hidden_size;
        let vocab = config.vocab_size;
        let outputs = vocab + config.durations.len();
        let row = |name: &str, len: usize| -> Result<Tensor<2>, WeightError> {
            Ok(load.vector_f32(name, len)?.reshape([1, len]))
        };
        let mut layers = Vec::with_capacity(config.num_decoder_layers);
        for l in 0..config.num_decoder_layers {
            let name = |kind: &str| format!("decoder.lstm.{kind}_l{l}");
            let b_ih = load.host_f32(&name("bias_ih"), &[4 * h])?;
            let b_hh = load.host_f32(&name("bias_hh"), &[4 * h])?;
            let bias: Vec<f32> = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
            layers.push(Lstm {
                input: load.tensor_f32(&name("weight_ih"), [4 * h, h])?,
                hidden: load.tensor_f32(&name("weight_hh"), [4 * h, h])?,
                bias: load.upload_f32(bias, [1, 4 * h]),
            });
        }
        Ok(Self {
            device: load.device.clone(),
            hidden: h,
            vocab,
            blank: config.blank_token_id,
            max_symbols: config.max_symbols_per_step,
            embed: load.tensor_f32("decoder.embedding.weight", [vocab, h])?,
            layers,
            projector: load.tensor_f32("decoder.decoder_projector.weight", [h, h])?,
            projector_bias: row("decoder.decoder_projector.bias", h)?,
            encoder_projector: load.tensor_f32("encoder_projector.weight", [h, d])?,
            encoder_projector_bias: row("encoder_projector.bias", h)?,
            head: load.tensor_f32("joint.head.weight", [outputs, h])?,
            head_bias: row("joint.head.bias", outputs)?,
            outputs,
        })
    }

    /// Projects encoder rows `[rows, d]` into the joint space `[rows, hidden]`. Pass the whole
    /// bucket-padded encoder output, so the product's shape does not vary with the audio.
    pub fn project(&self, encoded: Tensor<2>) -> Tensor<2> {
        linear(encoded, &self.encoder_projector) + self.encoder_projector_bias.clone()
    }

    /// Advances the LSTM by `token` and returns the projected prediction `[1, hidden]`.
    fn predict(&self, token: usize, state: &mut State) -> Tensor<2> {
        let h = self.hidden;
        let id =
            Tensor::<1, Int>::from_data(TensorData::new(vec![token as i32], [1]), &self.device);
        let mut x = self.embed.clone().select(0, id);
        for (l, layer) in self.layers.iter().enumerate() {
            let gates = linear(x, &layer.input)
                + linear(state.h[l].clone(), &layer.hidden)
                + layer.bias.clone();
            let gate = |i: usize| gates.clone().slice([0..1, i * h..(i + 1) * h]);
            let (input, forget, cell, output) = (
                sigmoid(gate(0)),
                sigmoid(gate(1)),
                gate(2).tanh(),
                sigmoid(gate(3)),
            );
            let c = forget * state.c[l].clone() + input * cell;
            let hidden = output * c.clone().tanh();
            state.c[l] = c;
            state.h[l] = hidden.clone();
            x = hidden;
        }
        if token != self.blank {
            state.tokens += 1;
        }
        linear(x, &self.projector) + self.projector_bias.clone()
    }

    /// The joint's choices for projected encoder rows against one prediction.
    /// Scores [`LOOKAHEAD`] rows, so every round trip has the same shapes.
    fn choose(&self, rows: Tensor<2>, prediction: &Tensor<2>) -> Vec<Choice> {
        let [n, _] = rows.dims();
        let logits = linear(relu(rows + prediction.clone()), &self.head) + self.head_bias.clone();
        let tokens = greedy(logits.clone().slice([0..n, 0..self.vocab]));
        let durations = logits.slice([0..n, self.vocab..self.outputs]).argmax(1);
        let durations = durations
            .into_data()
            .try_to_vec_as::<i64>()
            .expect("argmax indices");
        tokens
            .into_iter()
            .zip(durations)
            .map(|((token, probability), duration)| Choice {
                token: token as usize,
                logprob: probability.ln(),
                duration: duration as usize,
            })
            .collect()
    }

    /// Greedy TDT decoding of the first `frames` rows of projected encoder rows
    /// `[rows, hidden]`.
    pub fn decode(&self, projected: Tensor<2>, frames: usize) -> Vec<Emission> {
        let [_, h] = projected.dims();
        let padding = Tensor::zeros([LOOKAHEAD, h], &self.device);
        let projected = Tensor::cat(vec![projected, padding], 0);
        let zeros = || vec![Tensor::zeros([1, h], &self.device); self.layers.len()];
        let mut state = State {
            h: zeros(),
            c: zeros(),
            tokens: 0,
        };
        let mut prediction = self.predict(self.blank, &mut state);
        let mut window: Vec<Choice> = Vec::new();
        let mut start = 0;
        walk(frames, self.blank, self.max_symbols, |t, emitted| {
            if let Some(last) = emitted.last()
                && emitted.len() > state.tokens
            {
                prediction = self.predict(last.token, &mut state);
                window.clear();
            }
            if t < start || t >= start + window.len() {
                start = t;
                let rows = projected.clone().slice([t..t + LOOKAHEAD, 0..h]);
                window = self.choose(rows, &prediction);
            }
            window[t - start]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLANK: usize = 9;

    fn choice(token: usize, duration: usize) -> Choice {
        Choice {
            token,
            logprob: -0.5,
            duration,
        }
    }

    #[test]
    fn blanks_skip_ahead_and_tokens_advance_by_their_duration() {
        // frame 0: blank +2; frame 2: token 4 +0; frame 2 again: token 5 +3; frame 5: blank +0
        // (forced to 1); frame 6: token 1 +4 → past the end.
        let script = [
            (0, 0, choice(BLANK, 2)),
            (2, 0, choice(4, 0)),
            (2, 1, choice(5, 3)),
            (5, 2, choice(BLANK, 0)),
            (6, 2, choice(1, 4)),
        ];
        let mut steps = script.iter();
        let emitted = walk(8, BLANK, 10, |t, emitted| {
            let (frame, count, choice) = steps.next().expect("walk asked for too many steps");
            assert_eq!((t, emitted.len()), (*frame, *count));
            *choice
        });
        assert!(steps.next().is_none());
        let summary: Vec<_> = emitted
            .iter()
            .map(|e| (e.token, e.frame, e.frames))
            .collect();
        assert_eq!(summary, vec![(4, 2, 1), (5, 2, 3), (1, 6, 4)]);
    }

    #[test]
    fn repeated_zero_durations_are_capped_per_frame() {
        let emitted = walk(2, BLANK, 3, |t, _| {
            if t == 0 {
                choice(7, 0)
            } else {
                choice(BLANK, 1)
            }
        });
        assert_eq!(emitted.len(), 3);
        assert!(emitted.iter().all(|e| e.frame == 0));
    }
}
