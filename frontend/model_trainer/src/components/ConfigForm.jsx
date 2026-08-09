import {
  ACTIVATIONS,
  DATASETS,
  MAX_BATCH_SIZE,
  MAX_EPOCHS,
  MAX_LEARNING_RATE,
  MIN_BATCH_SIZE,
  MIN_LEARNING_RATE,
  OUTPUT_ACTIVATIONS,
  datasetByValue,
  lossFunctionsFor,
} from "../config";
import LayerEditor from "./LayerEditor";

export default function ConfigForm({ form, onChange, onSubmit, submitting, error }) {
  function set(field, value) {
    onChange({ ...form, [field]: value });
  }

  function setDataset(value) {
    const ds = datasetByValue(value);
    const validLosses = lossFunctionsFor(value).map((l) => l.value);
    const next = { ...form, dataset: value };
    // If the currently-selected loss isn't valid for the newly-picked
    // dataset's task type (e.g. switching to a regression dataset while
    // cross_entropy was selected), fall back to a valid one.
    if (!validLosses.includes(form.lossFunction)) {
      next.lossFunction = validLosses[0];
    }
    if (ds.taskType === "regression" && form.outputActivation === "softmax") {
      next.outputActivation = "linear";
    }
    onChange(next);
  }

  const ds = datasetByValue(form.dataset);
  const availableLosses = lossFunctionsFor(form.dataset);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <div className="field">
        <label className="field-label">Job name</label>
        <input
          type="text"
          value={form.jobName}
          onChange={(e) => set("jobName", e.target.value)}
          placeholder="e.g. iris-baseline-v1"
          maxLength={80}
          required
        />
      </div>

      <div className="field">
        <label className="field-label">Dataset</label>
        <select value={form.dataset} onChange={(e) => setDataset(e.target.value)}>
          {DATASETS.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label} ({d.taskType})
            </option>
          ))}
        </select>
        <p className="hint">
          Real dataset, loaded via scikit-learn on the worker &mdash; {ds.input} input features,{" "}
          {ds.taskType === "regression" ? "1 continuous target" : `${ds.output} classes`}.
        </p>
      </div>

      <LayerEditor layers={form.hiddenLayers} onChange={(v) => set("hiddenLayers", v)} />

      <div className="field-row">
        <div className="field">
          <label className="field-label">Hidden activation</label>
          <select value={form.activation} onChange={(e) => set("activation", e.target.value)}>
            {ACTIVATIONS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label className="field-label">Output activation</label>
          <select value={form.outputActivation} onChange={(e) => set("outputActivation", e.target.value)}>
            {OUTPUT_ACTIVATIONS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="field-row">
        <div className="field">
          <label className="field-label">Loss function</label>
          <select value={form.lossFunction} onChange={(e) => set("lossFunction", e.target.value)}>
            {availableLosses.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
          {ds.taskType === "regression" ? (
            <p className="hint">Cross-entropy needs class labels, so it's hidden for this regression dataset.</p>
          ) : null}
        </div>
        <div className="field">
          <label className="field-label">Epochs (max {MAX_EPOCHS})</label>
          <input
            type="number"
            min={1}
            max={MAX_EPOCHS}
            value={form.epochs}
            onChange={(e) => {
              const v = Math.min(Math.max(parseInt(e.target.value, 10) || 1, 1), MAX_EPOCHS);
              set("epochs", v);
            }}
          />
        </div>
      </div>

      <div className="field-row">
        <div className="field">
          <label className="field-label">Learning rate</label>
          <input
            type="number"
            step="0.0001"
            min={MIN_LEARNING_RATE}
            max={MAX_LEARNING_RATE}
            value={form.learningRate}
            onChange={(e) => set("learningRate", parseFloat(e.target.value) || MIN_LEARNING_RATE)}
          />
        </div>
        <div className="field">
          <label className="field-label">Batch size</label>
          <input
            type="number"
            step="1"
            min={MIN_BATCH_SIZE}
            max={MAX_BATCH_SIZE}
            value={form.batchSize}
            onChange={(e) => {
              const v = Math.min(Math.max(parseInt(e.target.value, 10) || MIN_BATCH_SIZE, MIN_BATCH_SIZE), MAX_BATCH_SIZE);
              set("batchSize", v);
            }}
          />
        </div>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <button type="submit" className="btn btn-primary" disabled={submitting}>
        {submitting ? "submitting…" : "start training run"}
      </button>
    </form>
  );
}
