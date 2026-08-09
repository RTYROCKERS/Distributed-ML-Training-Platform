import { useState } from "react";
import { BASE_URLS, submitTrainingJob } from "./api";
import ConfigForm from "./components/ConfigForm";
import NetworkDiagram from "./components/NetworkDiagram";
import NodeBadge from "./components/NodeBadge";
import TrainingMonitor from "./components/TrainingMonitor";
import { DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, datasetByValue, defaultHiddenLayer } from "./config";

const initialForm = {
  jobName: "",
  dataset: "iris",
  hiddenLayers: [defaultHiddenLayer(), { neurons: 32, dropout: 0.2 }],
  activation: "relu",
  outputActivation: "softmax",
  lossFunction: "cross_entropy",
  epochs: 10,
  learningRate: DEFAULT_LEARNING_RATE,
  batchSize: DEFAULT_BATCH_SIZE,
};

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [job, setJob] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      const payload = {
        job_name: form.jobName || "untitled-run",
        dataset: form.dataset,
        hidden_layers: form.hiddenLayers.map((l) => l.neurons),
        activation: form.activation,
        output_activation: form.outputActivation,
        loss_function: form.lossFunction,
        dropout: form.hiddenLayers.map((l) => l.dropout),
        epochs: form.epochs,
        learning_rate: form.learningRate,
        batch_size: form.batchSize,
      };
      const result = await submitTrainingJob(payload);
      const taskType = datasetByValue(form.dataset).taskType;
      setJob({ ...result, epochsTotal: form.epochs, taskType, job_name: payload.job_name });
    } catch (err) {
      setError(err.message || "Something went wrong submitting the job.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title-block" style={{ flexDirection: "column" }}>
          <span className="app-eyebrow">// active-active · fastapi + celery + redis</span>
          <h1 className="app-title">Distributed ML Training Platform</h1>
          <p className="app-subtitle">
            Configure a real neural network, train it on a real scikit-learn dataset, and
            watch epoch-by-epoch progress stream back live over a WebSocket &mdash; served
            by whichever of two independent backend nodes happens to answer.
          </p>
        </div>
        <div className="node-strip">
          <span className="node-strip-label">configured nodes</span>
          <div className="node-chip-row">
            {BASE_URLS.map((url) => (
              <NodeBadge key={url} baseUrl={url} />
            ))}
          </div>
        </div>
      </header>

      <div className="layout">
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">
              <strong>01</strong> · Network configuration
            </span>
          </div>
          <ConfigForm form={form} onChange={setForm} onSubmit={handleSubmit} submitting={submitting} error={error} />
        </div>

        <div>
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">
                <strong>02</strong> · Live architecture
              </span>
            </div>
            <NetworkDiagram
              dataset={form.dataset}
              hiddenLayers={form.hiddenLayers}
              activation={form.activation}
              outputActivation={form.outputActivation}
              pulseKey={job && job.job_id}
            />
          </div>

          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">
                <strong>03</strong> · Training run
              </span>
            </div>
            {job ? (
              <TrainingMonitor key={job.job_id} job={job} />
            ) : (
              <div className="empty-state">submit a run to see live epoch updates here</div>
            )}
          </div>
        </div>
      </div>

      <p className="footer-note">
        Training is real: an actual PyTorch model, sized from your architecture above, is
        trained with real gradient updates on a real scikit-learn dataset. What's
        distributed is the path a request takes: <code>POST /train</code> on either node
        &rarr; shared Redis (queue + job state) &rarr; whichever Celery worker picks it up
        &rarr; <code>WS /ws/jobs/&#123;job_id&#125;</code> on either node, streaming back
        real epoch metrics as they're computed. See the README for the full writeup.
      </p>
    </div>
  );
}
