// Client-side round-robin across the two live backend nodes. This is the
// "gateway" for the active-active setup: nothing fancy, just alternating
// through a small list of base URLs so real traffic actually lands on
// both Render and Railway. See README section "Active-active deployment".

const rawUrls = import.meta.env.VITE_API_URLS || "http://localhost:8000";
export const BASE_URLS = rawUrls
  .split(",")
  .map((u) => u.trim().replace(/\/$/, ""))
  .filter(Boolean);

let cursor = Math.floor(Math.random() * BASE_URLS.length);

export function nextBaseUrl() {
  const url = BASE_URLS[cursor % BASE_URLS.length];
  cursor += 1;
  return url;
}

// Used after submit to deliberately watch the job from the OTHER node
// (when more than one is configured) -- a visible, live demonstration
// that job state doesn't live in either node's memory.
export function otherBaseUrl(base) {
  if (BASE_URLS.length < 2) return base;
  return BASE_URLS.find((u) => u !== base) || base;
}

export function nodeLabel(baseUrl) {
  if (/localhost|127\.0\.0\.1/.test(baseUrl)) return "local";
  if (/render\.com/.test(baseUrl)) return "render";
  if (/railway\.app|up\.railway/.test(baseUrl)) return "railway";
  return baseUrl;
}

export async function submitTrainingJob(config) {
  const base = nextBaseUrl();
  let res;
  try {
    res = await fetch(`${base}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
  } catch {
    throw new Error(`Could not reach ${base}. Is the backend running?`);
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = formatDetail(body.detail);
    throw new Error(detail || `Request failed (${res.status})`);
  }

  const data = await res.json();
  return { ...data, apiBase: base };
}

export async function fetchJobStatus(apiBase, jobId) {
  const res = await fetch(`${apiBase}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job status request failed (${res.status})`);
  return res.json();
}

export function wsUrlFor(apiBase, jobId) {
  const wsBase = apiBase.replace(/^http/, "ws");
  return `${wsBase}/ws/jobs/${jobId}`;
}

// Fetches the trained model.pt as a Blob (kept in memory only -- the
// caller turns it into an object URL for a manual download link, and is
// responsible for revoking that URL once it's no longer needed).
export async function fetchModelBlob(apiBase, jobId) {
  const res = await fetch(`${apiBase}/jobs/${jobId}/model`);
  if (!res.ok) throw new Error(`Model download failed (${res.status})`);
  return res.blob();
}

function formatDetail(detail) {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return JSON.stringify(detail);
}
