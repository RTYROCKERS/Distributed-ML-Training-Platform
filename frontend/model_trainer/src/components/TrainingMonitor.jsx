import { useEffect, useRef, useState } from "react";
import { fetchModelBlob, otherBaseUrl, wsUrlFor } from "../api";
import EpochChart from "./EpochChart";
import NodeBadge from "./NodeBadge";

// Helper function to safely format numbers without throwing TypeErrors
const safeFixed = (val, digits = 3, fallback = "—") => {
  return typeof val === "number" && !isNaN(val) ? val.toFixed(digits) : fallback;
};

const safePercent = (val, digits = 1, fallback = "—") => {
  return typeof val === "number" && !isNaN(val) ? `${(val * 100).toFixed(digits)}%` : fallback;
};

export default function TrainingMonitor({ job }) {
  const [epochs, setEpochs] = useState([]);
  const [status, setStatus] = useState(job.status);
  const [wsNode, setWsNode] = useState(null);
  const [finalMetrics, setFinalMetrics] = useState(null);
  const [connectionError, setConnectionError] = useState(null);
  const [modelUrl, setModelUrl] = useState(null);
  const [modelError, setModelError] = useState(null);
  const socketRef = useRef(null);
  const modelUrlRef = useRef(null);

  const watchBase = otherBaseUrl(job.apiBase);

  async function loadModel(apiBase, jobId) {
    try {
      const blob = await fetchModelBlob(apiBase, jobId);
      const url = URL.createObjectURL(blob);
      modelUrlRef.current = url;
      setModelUrl(url);
    } catch (err) {
      setModelError(err.message || "Could not fetch the trained model.");
    }
  }

  useEffect(() => {
    setEpochs([]);
    setStatus("queued");
    setFinalMetrics(null);
    setConnectionError(null);
    setModelUrl(null);
    setModelError(null);
    modelUrlRef.current = null;

    const wsUrl = wsUrlFor(watchBase, job.job_id);
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      if (msg.type === "node") {
        setWsNode(watchBase);
      } else if (msg.type === "epoch") {
        setEpochs((prev) => [...prev, msg.data]);
        setStatus("running");
      } else if (msg.type === "status") {
        setStatus(msg.status);
        if (msg.status === "completed") {
          setFinalMetrics({ accuracy: msg.final_accuracy, loss: msg.final_loss });
        }
        if (msg.job && Array.isArray(msg.job.epochs_completed)) {
          setEpochs(msg.job.epochs_completed);
        }

        const modelReady = msg.model_ready || (msg.job && msg.job.model_ready);
        if (modelReady) {
          loadModel(watchBase, job.job_id);
        }
      } else if (msg.type === "error") {
        setConnectionError(msg.message);
      }
    };

    socket.onerror = () => {
      setConnectionError("Lost connection to the training socket.");
    };

    return () => {
      // Prevent "closed before connection established" warning on fast remounts
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.onopen = () => socket.close();
      } else {
        socket.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.job_id]);

  useEffect(() => {
    return () => {
      if (modelUrlRef.current) {
        URL.revokeObjectURL(modelUrlRef.current);
      }
    };
  }, []);

  const latest = epochs[epochs.length - 1];
  const metricLabel = job.taskType === "regression" ? "val R\u00B2" : "val accuracy";

  return (
    <div>
      <div className="status-row">
        <span className={`status-pill ${status}`}>
          {(status === "queued" || status === "running") && <span className="pulse-dot" />}
          {status}
        </span>
        <NodeBadge baseUrl={job.apiBase} label="submitted via" />
        <NodeBadge baseUrl={wsNode || watchBase} label="watching via" />
      </div>

      {connectionError ? <div className="error-banner">{connectionError}</div> : null}

      <div className="metric-grid">
        <div className="metric-box">
          <div className="metric-label">epoch</div>
          <div className="metric-value mono">
            {latest ? latest.epoch : 0}/{job.epochsTotal}
          </div>
        </div>
        <div className="metric-box">
          <div className="metric-label">val loss</div>
          <div className="metric-value mono">{safeFixed(latest?.val_loss, 3)}</div>
        </div>
        <div className="metric-box">
          <div className="metric-label">{metricLabel}</div>
          <div className="metric-value mono">
            {safePercent(latest?.val_accuracy, 1)}
          </div>
        </div>
      </div>

      {epochs.length > 0 ? (
        <>
          <EpochChart epochs={epochs} mode="loss" />
          <div style={{ height: 12 }} />
          <EpochChart epochs={epochs} mode="accuracy" />

          <table className="epoch-table">
            <thead>
              <tr>
                <th>epoch</th>
                <th>train loss</th>
                <th>val loss</th>
                <th>{metricLabel}</th>
              </tr>
            </thead>
            <tbody>
              {epochs.slice(-6).map((e, idx) => (
                <tr key={e.epoch ?? idx}>
                  <td>{e.epoch}</td>
                  <td>{safeFixed(e.train_loss, 3)}</td>
                  <td>{safeFixed(e.val_loss, 3)}</td>
                  <td>{safePercent(e.val_accuracy, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <div className="empty-state">waiting for the worker to pick up this job…</div>
      )}

      {finalMetrics ? (
        <p className="hint" style={{ marginTop: 12 }}>
          Run complete &mdash; final {metricLabel} {safePercent(finalMetrics.accuracy, 1)}, final loss{" "}
          {safeFixed(finalMetrics.loss, 3)}.
        </p>
      ) : null}

      {status === "completed" ? (
        <div style={{ marginTop: 8 }}>
          {modelUrl ? (
            <a
              className="btn"
              style={{ display: "inline-block", textDecoration: "none" }}
              href={modelUrl}
              download={`${job.job_name || "model"}.pt`}
            >
              Download model.pt
            </a>
          ) : modelError ? (
            <span className="error-banner">{modelError}</span>
          ) : (
            <span className="hint">fetching trained model&hellip;</span>
          )}
        </div>
      ) : null}
    </div>
  );
}