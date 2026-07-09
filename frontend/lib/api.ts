export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export interface ClassConfidence {
  label: string;
  confidence: number;
}

export interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  top_3: ClassConfidence[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  num_classes: number;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new ApiError("API health check failed", res.status);
  return res.json();
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    throw new ApiError(data?.detail || "Prediction failed.", res.status);
  }
  return data as PredictionResponse;
}

export function isHealthyLabel(label: string): boolean {
  return label.toLowerCase().includes("healthy");
}

export function formatLabel(label: string): string {
  return label.replace(/_/g, " ");
}
