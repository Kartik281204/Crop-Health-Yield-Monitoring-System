import type { Metadata } from "next";
import "./globals.css";

// Deliberately not using next/font/google here -- it fetches from Google
// Fonts at *build* time, which fails in network-restricted environments
// (this one included). System font stacks below cost zero extra requests
// and fit the diagnostic-instrument aesthetic fine on their own.

export const metadata: Metadata = {
  title: "Crop Diagnostic",
  description: "Upload a crop leaf photo, get a disease prediction back.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
