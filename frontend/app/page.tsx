"use client";

import { useState } from "react";
import StatusPill from "@/components/StatusPill";
import UploadZone from "@/components/UploadZone";
import ExampleGallery from "@/components/ExampleGallery";
import ResultCard from "@/components/ResultCard";
import { predictImage, PredictionResponse, ApiError } from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function selectFile(f: File, preview?: string) {
    setFile(f);
    setPreviewUrl(preview ?? URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(file);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-5 py-12">
      <div className="mb-2.5 flex items-center gap-2 font-mono text-xs uppercase tracking-[0.14em] text-leaf">
        <span className="inline-block h-[7px] w-[7px] rounded-full bg-leaf shadow-[0_0_8px_var(--color-leaf)]" />
        model online
      </div>

      <h1 className="mb-2 text-3xl font-semibold tracking-tight">
        Crop Diagnostic
      </h1>

      <p className="mb-8 max-w-[46ch] text-[14.5px] leading-relaxed text-text-muted">
        A CNN trained from scratch, checking leaf photos against 8 classes
        across 4 crops. Upload your own, or try a sample below. See the
        project README for training methodology and known limitations.
      </p>

      <div className="space-y-4 rounded-[10px] border border-border bg-surface p-5">
        <UploadZone
          previewUrl={previewUrl}
          fileName={file?.name ?? null}
          onFileSelected={(f) => selectFile(f)}
        />

        <button
          type="button"
          disabled={!file || loading}
          onClick={analyze}
          className="w-full rounded-[7px] bg-leaf py-[11px] text-sm font-semibold text-[#0e1309] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Analyzing\u2026" : file ? "Analyze leaf" : "Select an image first"}
        </button>

        {error && (
          <div className="rounded-lg border border-rust-dim bg-rust/10 px-3.5 py-3 font-mono text-[13px] text-rust">
            {"\u26a0"} {error}
          </div>
        )}

        <ExampleGallery
          disabled={loading}
          onPick={(f, preview) => selectFile(f, preview)}
        />
      </div>

      {result && (
        <div className="mt-4">
          <ResultCard result={result} />
        </div>
      )}

      <footer className="mt-10 flex justify-between font-mono text-[11.5px] text-text-muted">
        <span>crop-disease-detection &middot; phase 3</span>
        <StatusPill />
      </footer>
    </main>
  );
}
