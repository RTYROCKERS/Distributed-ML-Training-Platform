import { nodeLabel } from "../api";

export default function NodeBadge({ baseUrl, label }) {
  if (!baseUrl) return null;
  const node = nodeLabel(baseUrl);
  const dotClass = node === "render" ? "dot-render" : node === "railway" ? "dot-railway" : "dot-local";
  const textClass = node === "render" ? "node-render" : node === "railway" ? "node-railway" : "";

  return (
    <span className="node-chip">
      <span className={`node-dot ${dotClass}`} />
      {label ? `${label}: ` : ""}
      <span className={`mono ${textClass}`}>{node}</span>
    </span>
  );
}
