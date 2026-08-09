const WIDTH = 600;
const HEIGHT = 200;
const PAD_L = 34;
const PAD_R = 12;
const PAD_T = 10;
const PAD_B = 22;

function buildPath(values, maxVal) {
  if (values.length === 0) return "";
  const innerW = WIDTH - PAD_L - PAD_R;
  const innerH = HEIGHT - PAD_T - PAD_B;
  const n = values.length;
  return values
    .map((v, i) => {
      const x = PAD_L + (n === 1 ? innerW / 2 : (innerW * i) / (n - 1));
      const y = PAD_T + innerH - (Math.min(v, maxVal) / (maxVal || 1)) * innerH;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function EpochChart({ epochs, mode }) {
  const isAccuracy = mode === "accuracy";
  const series = isAccuracy
    ? [{ key: "val_accuracy", color: "var(--signal)", label: "val accuracy" }]
    : [
        { key: "train_loss", color: "var(--node-render)", label: "train loss" },
        { key: "val_loss", color: "var(--node-railway)", label: "val loss" },
      ];

  const allValues = series.flatMap((s) => epochs.map((e) => e[s.key]));
  const maxVal = isAccuracy ? 1 : Math.max(0.1, ...allValues, 0);

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${mode} chart`}>
        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = PAD_T + (HEIGHT - PAD_T - PAD_B) * (1 - t);
          return (
            <line
              key={t}
              x1={PAD_L}
              y1={y}
              x2={WIDTH - PAD_R}
              y2={y}
              stroke="var(--border-soft)"
              strokeWidth="1"
            />
          );
        })}
        {series.map((s) => (
          <path
            key={s.key}
            d={buildPath(
              epochs.map((e) => e[s.key]),
              maxVal
            )}
            fill="none"
            stroke={s.color}
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        <text x={PAD_L} y={HEIGHT - 6} fontSize="9" fill="var(--text-faint)" fontFamily="var(--font-mono)">
          epoch 1
        </text>
        <text
          x={WIDTH - PAD_R}
          y={HEIGHT - 6}
          fontSize="9"
          fill="var(--text-faint)"
          fontFamily="var(--font-mono)"
          textAnchor="end"
        >
          epoch {epochs.length}
        </text>
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span className="chart-legend-item" key={s.key}>
            <span className="legend-line" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
