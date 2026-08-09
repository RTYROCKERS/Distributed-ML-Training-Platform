import { useMemo } from "react";
import { MAX_VISIBLE_NEURONS, datasetByValue } from "../config";

const WIDTH = 640;
const HEIGHT = 300; // vertical space used for the neuron columns themselves
const VIEW_HEIGHT = HEIGHT + 46; // extra room below for layer labels
const NODE_R = 7;
const TOP_PAD = 24;
const BOTTOM_PAD = 24;

// Picks which neuron indices to actually draw for a layer of `count`
// neurons: everything if it's small, otherwise first 3 / ellipsis / last 3.
function visibleIndices(count) {
  if (count <= MAX_VISIBLE_NEURONS) {
    return Array.from({ length: count }, (_, i) => ({ index: i, ellipsis: false }));
  }
  const head = [0, 1, 2];
  const tail = [count - 3, count - 2, count - 1];
  return [
    ...head.map((i) => ({ index: i, ellipsis: false })),
    { index: -1, ellipsis: true },
    ...tail.map((i) => ({ index: i, ellipsis: false })),
  ];
}

function layerColumn(count) {
  const items = visibleIndices(count);
  const n = items.length;
  const usable = HEIGHT - TOP_PAD - BOTTOM_PAD;
  if (n === 1) {
    return items.map((item) => ({ ...item, y: HEIGHT / 2 }));
  }
  return items.map((item, i) => ({
    ...item,
    y: TOP_PAD + (usable * i) / (n - 1),
  }));
}

export default function NetworkDiagram({ dataset, hiddenLayers, activation, outputActivation, pulseKey }) {
  const ds = datasetByValue(dataset);

  const layers = useMemo(() => {
    const hidden = hiddenLayers.map((layer, i) => ({
      key: `hidden-${i}`,
      label: `H${i + 1}`,
      sublabel: `${layer.neurons}\u00A0·\u00A0${activation}`,
      count: Math.max(1, layer.neurons),
      kind: "hidden",
    }));
    return [
      { key: "input", label: "Input", sublabel: `${ds.input}`, count: ds.input, kind: "input" },
      ...hidden,
      { key: "output", label: "Output", sublabel: `${ds.output}\u00A0·\u00A0${outputActivation}`, count: ds.output, kind: "output" },
    ];
  }, [ds, hiddenLayers, activation, outputActivation]);

  const columns = layers.map((layer, i) => {
    const x = layers.length === 1 ? WIDTH / 2 : (WIDTH * i) / (layers.length - 1);
    const clampedX = Math.min(Math.max(x, 46), WIDTH - 46);
    return { ...layer, x: clampedX, nodes: layerColumn(layer.count) };
  });

  const colorFor = (kind) =>
    kind === "input" ? "var(--node-render)" : kind === "output" ? "var(--node-railway)" : "var(--signal)";

  return (
    <div>
      <svg className="diagram-svg" viewBox={`0 0 ${WIDTH} ${VIEW_HEIGHT}`} role="img" aria-label="Neural network architecture diagram">
        <g opacity="0.55">
          {columns.slice(0, -1).map((col, i) => {
            const next = columns[i + 1];
            return col.nodes
              .filter((n) => !n.ellipsis)
              .map((n) =>
                next.nodes
                  .filter((m) => !m.ellipsis)
                  .map((m) => (
                    <line
                      key={`${col.key}-${n.index}-${next.key}-${m.index}`}
                      x1={col.x}
                      y1={n.y}
                      x2={next.x}
                      y2={m.y}
                      stroke="var(--border)"
                      strokeWidth="1"
                    >
                      {pulseKey ? (
                        <animate
                          attributeName="stroke"
                          values="var(--border);var(--signal);var(--border)"
                          dur="1.1s"
                          begin={`${i * 0.12}s`}
                          keyTimes="0;0.5;1"
                        />
                      ) : null}
                    </line>
                  ))
              );
          })}
        </g>

        {columns.map((col) => (
          <g key={col.key}>
            {col.nodes.map((n, idx) =>
              n.ellipsis ? (
                <text
                  key={`ellipsis-${col.key}`}
                  x={col.x}
                  y={n.y + 4}
                  textAnchor="middle"
                  fontSize="14"
                  fill="var(--text-faint)"
                  fontFamily="var(--font-mono)"
                >
                  ⋮
                </text>
              ) : (
                <circle
                  key={`${col.key}-${idx}`}
                  cx={col.x}
                  cy={n.y}
                  r={NODE_R}
                  fill="var(--bg-raised)"
                  stroke={colorFor(col.kind)}
                  strokeWidth="1.6"
                />
              )
            )}
            <text
              x={col.x}
              y={HEIGHT - 8}
              textAnchor="middle"
              fontSize="11"
              fontFamily="var(--font-mono)"
              fill="var(--text-dim)"
              fontWeight="600"
            >
              {col.label}
            </text>
            <text
              x={col.x}
              y={HEIGHT + 6}
              textAnchor="middle"
              fontSize="9.5"
              fontFamily="var(--font-mono)"
              fill="var(--text-faint)"
            >
              {col.sublabel}
            </text>
          </g>
        ))}
      </svg>
      <div className="diagram-legend">
        <span className="diagram-legend-item">
          <span className="legend-swatch" style={{ background: "var(--node-render)" }} />
          input layer ({ds.label})
        </span>
        <span className="diagram-legend-item">
          <span className="legend-swatch" style={{ background: "var(--signal)" }} />
          hidden layers
        </span>
        <span className="diagram-legend-item">
          <span className="legend-swatch" style={{ background: "var(--node-railway)" }} />
          output layer
        </span>
      </div>
    </div>
  );
}
