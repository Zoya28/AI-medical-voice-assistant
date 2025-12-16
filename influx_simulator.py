"""
clean_and_run_simulator.py

Usage:
    python clean_and_run_simulator.py          → Runs with 10 regular users
    python clean_and_run_simulator.py --new    → Adds new random users along with regulars
"""

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import random, string, time, sys
import os

# --- CONFIG ---
URL = os.getenv("INFLUX_URL")
TOKEN = os.getenv("INFLUX_TOKEN")
ORG = os.getenv("INFLUX_ORG")
BUCKET = os.getenv("INFLUX_BUCKET")
WRITE_INTERVAL = 3  # seconds (data updates every 3 sec)
MAX_USERS = 10
# ----------------



REGULAR_USERS = [
    "OF36GM", "TK5FF6", "M9LY6H", "MPMQA6", "WR5EQT",
    "3KDOIZ", "JOTPOL", "CTV8U7", "2PNZGR", "LY9R2O"
]

def clear_measurement(client: InfluxDBClient, bucket: str, org: str, measurement: str):
    """
    Delete all points for a given measurement inside bucket.
    Uses a wide time window to ensure complete deletion.
    """
    delete_api = client.delete_api()
    start = "1970-01-01T00:00:00Z"
    stop = "2100-01-01T00:00:00Z"
    predicate = f'_measurement="{measurement}"'
    print(f"Deleting measurement '{measurement}' from bucket '{bucket}' ...")
    delete_api.delete(start, stop, predicate, bucket=bucket, org=org)
    print(f"Deleted measurement '{measurement}'")

def generate_userid():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def bpst_data(user):
    """
    Simulate realistic per-user 3-second step/motion data.
    Each user gets unique random behavior with logical dependencies.
    """
    random.seed(time.time_ns() + hash(user["UserId"]) % 10000)

    # Random walking pattern: most of the time user walks 0–8 steps, sometimes bursts up to 12
    if random.random() < 0.8:
        steps = random.randint(0, 8)
    else:
        steps = random.randint(8, 12)

    steps = int(steps)
    avg_step_length = random.uniform(0.6, 0.8)  # meters per step
    distance = round(steps * avg_step_length, 2)  # meters

    if steps == 0:
        speed = 0.0
    else:
        # distance (m) / time (s) → m/s
        speed = round(distance / WRITE_INTERVAL, 2)

    if speed > 8.0:
        speed = round(random.uniform(5.5, 8.0), 2)

    current_time = int(time.time())

    return (
        Point("bpst_data")
        .tag("UserId", user["UserId"])
        .field("ts", steps)
        .field("dc", distance)
        .field("sp", speed)
        .field("record_time", current_time)
    )

def dev_bpm_data(user):
    """
    Continuous vitals — correlated logically with movement intensity.
    """
    random.seed(time.time_ns() + hash(user["UserId"]) % 5000)

    activity_level = random.random()
    hr_base_min, hr_base_max = user["hr_range"]
    hr = random.randint(hr_base_min, hr_base_max)
    hr = int(hr + (activity_level * random.randint(5, 15)))

    hrv = int(max(25, 90 - hr / 2 + random.uniform(-5, 5)))
    dbp = int(70 + (hr - 70) * 0.15 + random.uniform(-2, 2))
    sbp = int(110 + (hr - 70) * 0.25 + random.uniform(-3, 3))
    spo2 = round(99 - (max(hr - 100, 0) * 0.05) + random.uniform(-0.2, 0.2), 1)
    cal = round(0.05 * hr + activity_level * random.uniform(0.2, 1.0), 1)

    return (
        Point("dev_bpm_data")
        .tag("UserId", user["UserId"])
        .field("HR", hr)
        .field("HRV", hrv)
        .field("DBP", dbp)
        .field("SBP", sbp)
        .field("SPO2", spo2)
        .field("cal", cal)
    )

def main():
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    try:
        # clear_measurement(client, BUCKET, ORG, "bpst_data")
        # clear_measurement(client, BUCKET, ORG, "dev_bpm_data")

        write_api = client.write_api(write_options=SYNCHRONOUS)

        users = [
            {
                "UserId": uid,
                "measurements": ["bpst_data", "dev_bpm_data"],
                "hr_range": [random.randint(60, 70), random.randint(85, 95)]
            }
            for uid in REGULAR_USERS
        ]

        if len(sys.argv) > 1 and sys.argv[1] == "--new":
            extra_count = 1
            for _ in range(extra_count):
                users.append({
                    "UserId": generate_userid(),
                    "measurements": ["bpst_data", "dev_bpm_data"],
                    "hr_range": [random.randint(60, 70), random.randint(85, 95)]
                })
            print(f"Added {extra_count} new users temporarily for this run.")

        print("\nSimulated Users:")
        for u in users:
            print(f"  - {u['UserId']} ({', '.join(u['measurements'])})")

        print("\nStarting Influx simulation (writes every 3 seconds)...")
        while True:
            points = []
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            for user in users:
                if "bpst_data" in user["measurements"]:
                    points.append(bpst_data(user))
                if "dev_bpm_data" in user["measurements"]:
                    points.append(dev_bpm_data(user))

            if points:
                write_api.write(bucket=BUCKET, org=ORG, record=points)
                print(f"[{timestamp_str}] Wrote {len(points)} points for {len(users)} users")

            time.sleep(WRITE_INTERVAL)

    except KeyboardInterrupt:
        print("\nSimulator stopped by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        client.close()
        print("InfluxDB client closed. Bye.")

if __name__ == "__main__":
    main()
