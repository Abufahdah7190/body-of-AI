"""Filter detections and estimate visual approach from a camera stream."""

HANDHELD = {
    "cup", "bottle", "cell phone", "book", "remote", "scissors", "mouse",
    "keyboard", "laptop", "backpack", "handbag", "sports ball", "banana",
    "apple", "orange", "fork", "knife", "spoon", "bowl", "toothbrush",
    "hair drier",
}

def display_objects(detections):
    persons = [
        row for row in detections
        if row.get("name") == "person" and row.get("confidence", 0) >= 0.35
    ]
    visible = []
    for row in detections:
        minimum = 0.4 if row.get("custom") is True else 0.6
        if not (
            (row.get("name") in HANDHELD or row.get("custom") is True)
            and row.get("confidence", 0) >= minimum
        ):
            continue
        # This experimental charger model sometimes labels a large person box
        # as a charger. Hide that known false-positive shape.
        if row.get("custom") is True and row.get("name") == "charger" and _covers_person(row, persons):
            continue
        visible.append(row)
    return visible

def _covers_person(row,persons):
    box=row.get('box')
    if not box or len(box)!=4:
        return False
    area=max(0,box[2]-box[0])*max(0,box[3]-box[1])
    if area<.55:
        return False
    for person in persons:
        p=person.get('box')
        if not p or len(p)!=4:
            continue
        person_area=max(0,p[2]-p[0])*max(0,p[3]-p[1])
        intersection=max(0,min(box[2],p[2])-max(box[0],p[0]))*max(0,min(box[3],p[3])-max(box[1],p[1]))
        if person_area and intersection/person_area>=.8:
            return True
    return False


class ApproachTracker:
    """Mark likely visual approach when a tracked, pretrained object grows."""

    def __init__(self):
        self.tracks = []

    def update(self, rows, now):
        current = []
        next_tracks = []
        used = set()
        for row in rows:
            # The charger model is not reliable enough to drive a proximity alert.
            if row.get("custom") is True:
                current.append({**row, "approaching": False})
                continue

            box = row.get("box")
            if not box or len(box) != 4:
                current.append({**row, "approaching": False})
                continue

            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            match = None
            match_index = None
            best_overlap = 0.25
            for index, track in enumerate(self.tracks):
                if index in used or track["name"] != row["name"]:
                    continue
                score = overlap(track["box"], box)
                if score > best_overlap:
                    match, match_index, best_overlap = track, index, score

            if match is None:
                track = {
                    "name": row["name"],
                    "box": box,
                    "first_seen": now,
                    "baseline_area": area,
                    "last_area": area,
                    "growth_hits": 0,
                    "approaching_until": 0,
                }
            else:
                used.add(match_index)
                track = dict(match)
                if now - track["first_seen"] > 2.5:
                    track["first_seen"] = now
                    track["baseline_area"] = area
                    track["growth_hits"] = 0
                    track["approaching_until"] = 0
                else:
                    if area > track["last_area"] * 1.08:
                        track["growth_hits"] += 1
                    elif area < track["last_area"] * 0.96:
                        track["growth_hits"] = max(0, track["growth_hits"] - 1)
                    track["baseline_area"] = min(track["baseline_area"], area)
                    if (
                        now - track["first_seen"] >= 1.0
                        and area >= track["baseline_area"] * 1.55
                        and track["growth_hits"] >= 2
                    ):
                        track["approaching_until"] = now + 1.25

                track["box"] = box
                track["last_area"] = area

            current.append({**row, "approaching": now < track["approaching_until"]})
            track["name"] = row["name"]
            track["box"] = box
            track["last_area"] = area
            next_tracks.append(track)

        self.tracks = [track for track in next_tracks if now - track["first_seen"] <= 2.5]
        return current

def overlap(a,b):
    l=max(a[0],b[0]);t=max(a[1],b[1]);r=min(a[2],b[2]);bot=min(a[3],b[3]);inter=max(0,r-l)*max(0,bot-t)
    union=max(0,(a[2]-a[0])*(a[3]-a[1]))+max(0,(b[2]-b[0])*(b[3]-b[1]))-inter
    return inter/union if union>0 else 0
class StableObjects:
    def __init__(self):self.previous=[];self.last=None
    def update(self,rows,now):
        if self.last is None or now-self.last>1.5:self.previous=[]
        current=[]
        for row in rows:
            minimum=.4 if row.get('custom') is True else .6
            if row.get('confidence',0)<minimum:continue
            count=1+max([p['count'] for p in self.previous if p['row']['name']==row['name'] and overlap(p['row']['box'],row['box'])>=.3] or [0])
            current.append({'row':row,'count':count})
        self.previous=current;self.last=now
        return [p['row'] for p in current if p['count']>=3]
