import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "HDFC MF Assistant",
    description: "AI-powered factual assistant for HDFC Mutual Funds",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className="antialiased font-sans">
                {children}
            </body>
        </html>
    );
}
