"use client";

import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  previewUrl: string | null;
  fileName: string | null;
  onFileSelected: (file: File) => void;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function UploadZone({
  previewUrl,
  fileName,
  onFileSelected,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!ACCEPTED_TYPES.includes(file.type)) return;
      onFileSelected(file);
    },
    [onFileSelected],
  );

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      className={`relative cursor-pointer rounded-lg border-[1.5px] border-dashed p-9 text-center transition-colors ${
        dragging
          ? "border-leaf-dim bg-leaf/5"
          : "border-border hover:border-leaf-dim hover:bg-leaf/[0.03]"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {previewUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={previewUrl}
          alt="preview"
          className="mx-auto max-h-56 rounded-md"
        />
      ) : null}

      <p className="mt-3 font-mono text-[12.5px] text-text-muted">
        {fileName ? (
          <>
            <strong className="font-medium text-leaf">{fileName}</strong>{" "}
            &middot; click to replace
          </>
        ) : (
          <>
            <strong className="font-medium text-leaf">Click to upload</strong>{" "}
            or drag a leaf photo here &middot; JPG / PNG / WebP, up to 8MB
          </>
        )}
      </p>
    </div>
  );
}
