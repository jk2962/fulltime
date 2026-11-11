import type { Metadata } from "next";
import { Archivo } from "next/font/google";
import "./globals.css";

// One family. Its width axis gives the condensed team codes.
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  axes: ["wdth"],
});

export const metadata: Metadata = {
  title: "Fulltime · Premier League forecasts",
  description: "Premier League match forecasts, locked before kickoff and scored against the betting market.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${archivo.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">{children}</body>
    </html>
  );
}
