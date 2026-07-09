"use client";

import { formatLabel } from "@/lib/api";

const SAMPLES = [
  "Apple_healthy",
  "Apple_Black_rot",
  "Corn_healthy",
  "Corn_Common_rust",
  "Potato_healthy",
  "Potato_Early_blight",
  "Tomato_healthy",
  "Tomato_Late_blight",
];

interface ExampleGalleryProps {
  onPick: (file: File, previewUrl: string) => void;
  disabled?: boolean;
}

export default function ExampleGallery({ onPick, disabled }: ExampleGalleryProps) {
  async function pick(name: string) {
    const url = `/samples/${name}.jpg`;
    const res = await fetch(url);
    const blob = await res.blob();
    const file = new File([blob], `${name}.jpg`, { type: "image/jpeg" });
    onPick(file, url);
  }

  return (
    <div>
      <div className="mb-2.5 font-mono text-[11px] uppercase tracking-[0.1em] text-text-muted">
        or try a sample
      </div>
      <div className="grid grid-cols-4 gap-2">
        {SAMPLES.map((name) => (
          <button
            key={name}
            type="button"
            disabled={disabled}
            onClick={() => pick(name)}
            title={formatLabel(name)}
            className="group relative aspect-square overflow-hidden rounded-md border border-border transition-colors hover:border-leaf-dim disabled:cursor-not-allowed disabled:opacity-40"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`/samples/${name}.jpg`}
              alt={formatLabel(name)}
              className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105"
            />
          </button>
        ))}
      </div>
    </div>
  );
}
