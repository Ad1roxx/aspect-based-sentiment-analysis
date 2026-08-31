/**
 * Client for the ABSA API.
 *
 * The base URL comes from VITE_API_URL so the same build can point at a local
 * uvicorn, a container, or a deployed host. Hardcoding localhost is the classic
 * way to ship a frontend that only works on the machine it was written on.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export const MAX_LENGTH = 5000;

/** Thrown for anything the user should see a readable message about. */
export class ApiError extends Error {
  constructor(message, { status = null, unreachable = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.unreachable = unreachable;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    // fetch only rejects on network-level failure, so this branch means the
    // server is not reachable at all — a different problem from a 4xx/5xx, and
    // the user needs a different message for it.
    throw new ApiError(
      `Cannot reach the API at ${BASE_URL}. Is the server running?`,
      { unreachable: true },
    );
  }

  if (!response.ok) {
    // FastAPI puts the reason in `detail`, which is either a string (our
    // explicit HTTPExceptions) or a list of field errors (Pydantic validation).
    let detail = null;
    try {
      const body = await response.json();
      detail = Array.isArray(body.detail)
        ? body.detail.map((d) => d.msg).join("; ")
        : body.detail;
    } catch {
      detail = null;
    }
    throw new ApiError(detail ?? `Request failed (${response.status})`, {
      status: response.status,
    });
  }

  return response.json();
}

export function predict(text, explain = false) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ text, explain }),
  });
}

export function modelInfo() {
  return request("/model-info");
}
