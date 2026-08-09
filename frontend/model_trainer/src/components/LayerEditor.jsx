import { MAX_DROPOUT, MAX_HIDDEN_LAYERS, MAX_NEURONS_PER_LAYER, defaultHiddenLayer } from "../config";

export default function LayerEditor({ layers, onChange }) {
  function updateLayer(index, patch) {
    const next = layers.map((l, i) => (i === index ? { ...l, ...patch } : l));
    onChange(next);
  }

  function addLayer() {
    if (layers.length >= MAX_HIDDEN_LAYERS) return;
    onChange([...layers, defaultHiddenLayer()]);
  }

  function removeLayer(index) {
    onChange(layers.filter((_, i) => i !== index));
  }

  return (
    <div className="field">
      <label className="field-label">
        Hidden layers ({layers.length}/{MAX_HIDDEN_LAYERS})
      </label>

      {layers.map((layer, i) => (
        <div className="layer-card" key={i}>
          <div className="layer-card-head">
            <span className="layer-card-title">LAYER {i + 1}</span>
            <button
              type="button"
              className="icon-btn"
              onClick={() => removeLayer(i)}
              disabled={layers.length <= 1}
              aria-label={`Remove layer ${i + 1}`}
              title="Remove layer"
            >
              &minus;
            </button>
          </div>

          <div className="field-row">
            <div className="field" style={{ marginBottom: 0 }}>
              <label className="field-label">Neurons (max {MAX_NEURONS_PER_LAYER})</label>
              <input
                type="number"
                min={1}
                max={MAX_NEURONS_PER_LAYER}
                value={layer.neurons}
                onChange={(e) => {
                  const v = clamp(parseInt(e.target.value, 10) || 1, 1, MAX_NEURONS_PER_LAYER);
                  updateLayer(i, { neurons: v });
                }}
              />
            </div>
          </div>

          <div className="field" style={{ marginBottom: 0, marginTop: 10 }}>
            <label className="field-label">Dropout rate</label>
            <div className="slider-row">
              <input
                type="range"
                min={0}
                max={MAX_DROPOUT}
                step={0.05}
                value={layer.dropout}
                onChange={(e) => updateLayer(i, { dropout: parseFloat(e.target.value) })}
              />
              <span className="slider-value mono">{layer.dropout.toFixed(2)}</span>
            </div>
          </div>
        </div>
      ))}

      <button
        type="button"
        className="btn btn-add-layer"
        onClick={addLayer}
        disabled={layers.length >= MAX_HIDDEN_LAYERS}
      >
        + add hidden layer
      </button>
    </div>
  );
}

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max);
}
