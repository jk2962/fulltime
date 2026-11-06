import { Suspense } from "react";
import Link from "next/link";
import { ErrorNote } from "@/components/ErrorNote";
import { Header, SHELL } from "@/components/Header";
import { TeamChip } from "@/components/TeamChip";
import { getMatches, getMatchweeks } from "@/lib/api";
import { percent, ukDateTime } from "@/lib/format";
import { OUTCOMES, modelProbs, summarize } from "@/lib/matches";

export default function Home() {
  return (
    <>
      <Header />
      <main className={`${SHELL} py-10`}>
        <section className="max-w-prose">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">Premier League forecasts, locked before kickoff.</h1>
          <p className="mt-4 text-lg text-muted">
            Win, draw and loss probabilities for every match, and the whole season simulated 10,000 times. Each forecast is
            committed before the first whistle and scored against the betting market afterwards.
          </p>
        </section>
        <Suspense fallback={<p className="mt-10 text-muted">Loading…</p>}>
          <Latest />
        </Suspense>
        <HowItWorks />
      </main>
    </>
  );
}

async function Latest() {
  const result = await getMatchweeks();
  const locked = result.ok ? result.data.filter((m) => m.matchesLockedAt || m.tableLockedAt) : [];
  if (!result.ok || locked.length === 0) {
    return (
      <div className="mt-10">
        <ErrorNote message={result.ok ? "No matchweek has locked predictions yet." : result.error} />
      </div>
    );
  }

  // The week to feature: the first locked one not yet complete, else the latest.
  const current = locked.find((m) => m.status !== "complete") ?? locked[locked.length - 1];
  const weeks = await Promise.all(locked.map((m) => getMatches(m.matchweek)));
  const byWeek = new Map(locked.map((m, i) => [m.matchweek, weeks[i]]));
  const thisWeek = byWeek.get(current.matchweek);
  const fixtures = thisWeek?.ok ? [...thisWeek.data].sort((a, b) => Date.parse(a.kickoff) - Date.parse(b.kickoff)) : [];
  // Retrospective weeks are excluded above, so this is the locked record only.
  const season = summarize(weeks.flatMap((w) => (w.ok ? w.data : []))).rps;
  const n = current.matchweek;

  return (
    <>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link href={`/mw/${n}/matches`} className="bg-ink px-5 py-3 font-semibold text-surface hover:opacity-90">
          MW {n} match forecasts
        </Link>
        <Link href={`/mw/${n}`} className="border border-ink px-5 py-3 font-semibold hover:bg-surface">
          Season forecast
        </Link>
        <Link href="/mw/1" className="px-5 py-3 font-medium text-muted underline underline-offset-4 hover:text-ink">
          Start from MW 1
        </Link>
      </div>

      {season && (
        <section aria-labelledby="record" className="mt-12">
          <h2 id="record" className="text-lg font-bold">
            Season so far
          </h2>
          <p className="text-sm text-muted">
            Mean RPS over {season.matches} locked, played matches. Lower is better.
          </p>
          <dl className="mt-4 grid grid-cols-3 gap-px border border-rule bg-rule">
            {[
              ["Model", season.model],
              ["Market", season.market],
              ["Naive", season.naive],
            ].map(([label, value]) => (
              <div key={label} className="bg-surface p-4">
                <dt className="text-sm text-muted">{label}</dt>
                <dd className="text-2xl font-bold">{(value as number).toFixed(3)}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {fixtures.length > 0 && (
        <section aria-labelledby="week" className="mt-12">
          <div className="flex items-baseline justify-between gap-4">
            <h2 id="week" className="text-lg font-bold">
              Matchweek {n}
            </h2>
            <Link href={`/mw/${n}/matches`} className="text-sm font-medium underline underline-offset-4">
              All details
            </Link>
          </div>
          {current.matchesLockedAt && (
            <p className="text-sm text-muted">
              Locked <time dateTime={current.matchesLockedAt}>{ukDateTime(current.matchesLockedAt)}</time>
            </p>
          )}
          <ul className="mt-4 divide-y divide-rule border border-rule bg-surface">
            {fixtures.map((m) => {
              const p = modelProbs(m);
              return (
                <li key={m.id} className="grid grid-cols-[1fr_auto] items-center gap-4 px-4 py-3">
                  <span className="flex items-center gap-2">
                    <TeamChip code={m.home} />
                    <span className="text-muted">v</span>
                    <TeamChip code={m.away} />
                    {m.result && (
                      <span className="ml-2 text-sm font-semibold">
                        {m.result.home} – {m.result.away}
                      </span>
                    )}
                  </span>
                  <span
                    className="grid grid-cols-3 gap-3 text-right text-sm font-medium"
                    aria-label={`Home ${percent(p.home)}, draw ${percent(p.draw)}, away ${percent(p.away)}`}
                  >
                    {OUTCOMES.map((o) => (
                      <span key={o} className="w-10">
                        {percent(p[o])}
                      </span>
                    ))}
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 text-right text-xs text-muted">Home · Draw · Away</p>
        </section>
      )}
    </>
  );
}

function HowItWorks() {
  const steps = [
    ["Model", "A Poisson goals model rates every team's attack and defence from the last two seasons of results."],
    ["Lock", "Forecasts are committed to a public repository before kickoff and never edited afterwards."],
    ["Score", "Each forecast is scored with the ranked probability score next to the closing betting odds."],
  ];
  return (
    <section aria-labelledby="how" className="mt-12 border-t border-rule pt-8">
      <h2 id="how" className="text-lg font-bold">
        How it works
      </h2>
      <ol className="mt-4 grid gap-6 sm:grid-cols-3">
        {steps.map(([title, body], i) => (
          <li key={title}>
            <p className="text-sm font-semibold text-muted">0{i + 1}</p>
            <p className="mt-1 font-bold">{title}</p>
            <p className="mt-1 text-sm leading-relaxed">{body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
