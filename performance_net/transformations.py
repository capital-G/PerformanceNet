from typing import NamedTuple, Optional

import mido
import numpy as np
import pandas as pd


class QuantizedValue(NamedTuple):
    quantized_value: int
    scaled_value: float


class PianoRoll:
    def __init__(self, midi_file_path: str):
        self._midi_file_path = midi_file_path
        self.midi_file = mido.MidiFile(self._midi_file_path)
        self.tempo = self.get_tempo()

    def get_tempo(self) -> Optional[float]:
        for msg in self.midi_file:
            if msg.type == "set_tempo":
                return mido.tempo2bpm(msg.tempo)
        return None

    def events(self, hold_notes_with_pedal: bool = False) -> pd.DataFrame:
        if hold_notes_with_pedal:
            raise NotImplemented

        events = []
        time = 0.0
        for msg in self.midi_file:
            time += msg.time
            if any([x == msg.type for x in ["note_on", "note_off"]]):
                events.append(
                    {
                        "note": msg.note,
                        "velocity": msg.velocity if msg.type == "note_on" else 0,
                        "time": time,
                    }
                )
            # pedal
            elif msg.is_cc(64):
                events.append(
                    {
                        "note": -1,
                        "velocity": msg.value,
                        "time": time,
                    }
                )
        df_events = pd.DataFrame(events)

        return df_events

    def performance_vector_one_hot(
        self,
        num_time_slots: int = 125,
        num_velocities: int = 32,
        include_pedal: bool = True,
    ) -> np.ndarray:
        events_df = self.events()
        vectors = []
        events_df["time_delta"] = events_df["time"].diff(1).fillna(0.0)

        performance_vector_factory = PerformanceVectorFactory(
            num_notes=88,
            num_velocities=num_velocities,
            num_time=num_time_slots,
            include_pedal=include_pedal,
        )

        for _, event in events_df.iterrows():
            if event["time_delta"] >= 0.01:
                # only add time vector if time actually moves
                vectors.append(
                    performance_vector_factory.time_vector(event["time_delta"])
                )
            if event["velocity"] > 0:
                vectors.append(
                    performance_vector_factory.velocity_vector(event["velocity"])
                )
                vectors.append(performance_vector_factory.note_on_vector(event["note"]))
            else:
                vectors.append(
                    performance_vector_factory.note_off_vector(event["note"])
                )

        return np.array(vectors, dtype=np.float32)

    def __repr__(self) -> str:
        return f"PianoRoll({self._midi_file_path})"


def quantize(
    value, in_min, in_max, steps: int, out_min=0.0, out_max=1.0
) -> QuantizedValue:
    normalized_value = (value - in_min) / (in_max - in_min)
    scaled_value = np.floor(normalized_value * steps) / steps
    quantized_value = (scaled_value * (out_max - out_min)) + out_min
    return QuantizedValue(
        quantized_value=int(quantized_value),
        scaled_value=scaled_value,
    )


class PerformanceVectorFactory:
    """Convenience method which allows us to create and read a performance vector
    in a programmatic manner by encapsulating its state in a class.
    """

    def __init__(
        self,
        num_notes: int = 88,
        num_time=125,
        num_velocities: int = 32,
        include_pedal: bool = True,
        midi_note_offset: int = 21,
    ) -> None:
        self.num_note_on = num_notes
        self.num_note_off = num_notes
        self.num_time = num_time
        self.num_velocities = num_velocities
        self.include_pedal = include_pedal
        if self.include_pedal:
            self.num_note_on += 1
            self.num_note_off += 1
        self.midi_note_offset = midi_note_offset

    def _blank_vector(self) -> np.ndarray:
        return np.zeros(
            shape=(
                self.num_note_on
                + self.num_note_off
                + self.num_time
                + self.num_velocities
            )
        )

    def velocity_vector(self, velocity: int) -> np.ndarray:
        velocity = np.clip(velocity, 0, 127)
        quantized_velocity_num = int(
            np.round(
                quantize(
                    value=velocity,
                    in_min=0,
                    in_max=127,
                    steps=self.num_velocities,
                ).scaled_value
                * (self.num_velocities - 1)
            )
        )
        vector = self._blank_vector()
        vector[
            self.num_note_on
            + self.num_note_off
            + self.num_time
            + quantized_velocity_num
        ] = 1.0
        return vector

    def time_vector(self, time: float) -> np.ndarray:
        # time in s
        time = np.clip(time, 0, 1.0)
        quantized_time_num = int(
            np.round(
                quantize(
                    value=time,
                    in_min=0,
                    in_max=1.0,
                    steps=self.num_time,
                ).scaled_value
                * (self.num_time - 1)
            )
        )
        vector = self._blank_vector()
        vector[self.num_note_on + self.num_note_off + quantized_time_num] = 1.0
        return vector

    def note_on_vector(self, note_on: int) -> np.ndarray:
        if note_on == -1:
            if not self.include_pedal:
                raise Exception("Encountered pedal on value in non-pedal factory")
            note_on = self.num_note_on
        else:
            note_on = np.clip(note_on - self.midi_note_offset, 0, self.num_note_on)
        vector = self._blank_vector()
        vector[int(note_on)] = 1.0
        return vector

    def note_off_vector(self, note_off: int) -> np.ndarray:
        if note_off == -1:
            if not self.include_pedal:
                raise Exception("Encountered pedal off value in non-pedal factory")
            note_off = self.num_note_off
        else:
            note_off = np.clip(note_off - self.midi_note_offset, 0, self.num_note_off)
        vector = self._blank_vector()
        vector[self.num_note_on + int(note_off)] = 1.0
        return vector

    def reverse_vector(self, vector: np.ndarray):
        raise NotImplemented
