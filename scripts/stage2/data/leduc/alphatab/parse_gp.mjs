/**
 * Parse a Guitar Pro file using alphaTab and output:
 *   1. JSON with note data (pitch, string, fret, timing) from the score model
 *   2. MIDI file via alphaTab's built-in MidiFileGenerator (handles repeats, tempo, ties, etc.)
 *
 */

import * as alphaTab from "@coderline/alphatab";
import { writeFileSync, readFileSync } from "fs";

const args = process.argv.slice(2);
const inputPath = args.find((a) => !a.startsWith("--"));
const pretty = args.includes("--pretty");
const infoOnly = args.includes("--info");
const midiIdx = args.indexOf("--midi");
const midiPath = midiIdx !== -1 ? args[midiIdx + 1] : null;

if (!inputPath) {
  console.error(
    "Usage: node parse_gp.mjs <file.gp> [--pretty] [--info] [--midi out.mid]"
  );
  process.exit(1);
}

const fileData = readFileSync(inputPath);
const settings = new alphaTab.Settings();
const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(
  new Uint8Array(fileData),
  settings
);

if (infoOnly) {
  console.error(`Title: ${score.title}`);
  console.error(`Artist: ${score.artist}`);
  console.error(`Tempo: ${score.tempo}`);
  console.error(`Tracks: ${score.tracks.length}`);
  for (const track of score.tracks) {
    console.error(
      `  ${track.name}: ${track.staves.length} staves, tuning=${track.staves[0]?.tuning}`
    );
  }
  console.error(`MasterBars: ${score.masterBars.length}`);
  process.exit(0);
}

// --- Generate MIDI using alphaTab's built-in generator ---
if (midiPath) {
  const midiFile = new alphaTab.midi.MidiFile();
  const generator = new alphaTab.midi.MidiFileGenerator(
    score,
    settings,
    new alphaTab.midi.AlphaSynthMidiFileHandler(midiFile)
  );
  generator.generate();
  const midiBytes = midiFile.toBinary();
  writeFileSync(midiPath, midiBytes);
  console.error(`MIDI written: ${midiPath} (${midiBytes.length} bytes)`);
}

// --- Extract note data with timing from the score model ---
// Use alphaTab's MidiUtils for accurate tick-to-time conversion
const ticksPerBeat = 960;
const notes = [];

// Build tempo map from master bars for tick->seconds conversion
const tempoMap = []; // [{tick, bpm}]
let barTick = 0;
for (const mb of score.masterBars) {
  const barTicks =
    (ticksPerBeat * 4 * mb.timeSignatureNumerator) /
    mb.timeSignatureDenominator;

  if (mb.tempoAutomation) {
    tempoMap.push({ tick: barTick, bpm: mb.tempoAutomation.value });
  }
  barTick += barTicks;
}
if (tempoMap.length === 0) {
  tempoMap.push({ tick: 0, bpm: score.tempo });
}

function tickToSeconds(tick) {
  let seconds = 0;
  let prevTick = 0;
  let bpm = tempoMap[0].bpm;

  for (const tp of tempoMap) {
    if (tp.tick > tick) break;
    seconds += ((tp.tick - prevTick) / ticksPerBeat) * (60 / bpm);
    prevTick = tp.tick;
    bpm = tp.bpm;
  }
  seconds += ((tick - prevTick) / ticksPerBeat) * (60 / bpm);
  return seconds;
}

// Walk score: first track, first staff
const mainTrack = score.tracks[0];
const mainStaff = mainTrack.staves[0];

barTick = 0;
for (const bar of mainStaff.bars) {
  const masterBar = score.masterBars[bar.index];
  const barTicks =
    (ticksPerBeat * 4 * masterBar.timeSignatureNumerator) /
    masterBar.timeSignatureDenominator;

  for (const voice of bar.voices) {
    for (const beat of voice.beats) {
      if (beat.notes.length === 0) continue;

      const absTick = barTick + beat.playbackStart;

      for (const note of beat.notes) {
        if (note.isTieDestination) continue;

        let durationTicks = beat.playbackDuration;
        let tiedNote = note.tieDestination;
        while (tiedNote) {
          durationTicks += tiedNote.beat.playbackDuration;
          tiedNote = tiedNote.tieDestination;
        }

        // alphaTab counts strings from bottom (1=low E), flip to guitar convention (1=high E)
        const nStrings = mainStaff.tuning.length;
        notes.push({
          pitch: note.realValue,
          string: nStrings - note.string + 1,
          fret: note.fret,
          start: Math.round(tickToSeconds(absTick) * 10000) / 10000,
          end:
            Math.round(tickToSeconds(absTick + durationTicks) * 10000) / 10000,
        });
      }
    }
  }

  barTick += barTicks;
}

notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);

// Effective starting tempo: what alphaTab actually used for tickToSeconds.
// score.tempo is the file-level default (often 120 for Leduc); the first
// tempoMap entry reflects bar-0 tempoAutomations and is what the timing reflects.
const effectiveTempo = tempoMap[0]?.bpm ?? score.tempo;

const output = {
  title: score.title,
  artist: score.artist,
  tempo: effectiveTempo,
  tracks: score.tracks.map((t) => ({
    name: t.name,
    tuning: t.staves[0]?.tuning || [],
    n_staves: t.staves.length,
  })),
  n_bars: score.masterBars.length,
  n_notes: notes.length,
  notes,
};

process.stdout.write(
  (pretty ? JSON.stringify(output, null, 2) : JSON.stringify(output)) + "\n"
);
