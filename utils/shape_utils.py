import math
import numpy as np
from PIL import ImageDraw

### GEOMETRY HELPER FUNCTIONS

# --- spline helper: closed Catmull–Rom (uniform) ---
def _catmull_rom(p0, p1, p2, p3, t: float):
    # Uniform Catmull–Rom spline
    t2, t3 = t * t, t * t * t
    x = 0.5 * (
        (2 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return (x, y)

def closed_catmull_rom(points, samples_per_seg: int = 25):
    """
    points: list[(x,y)] ordered around the shape (radially sorted).
    returns: dense list[(x,y)] that forms a smooth CLOSED curve.
    """
    n = len(points)
    if n < 3:
        return points

    curve = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i % n]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        for s in range(samples_per_seg):
            t = s / float(samples_per_seg)
            curve.append(_catmull_rom(p0, p1, p2, p3, t))
    return curve

def open_catmull_rom(points, samples_per_seg: int = 25):
    """
    points: list[(x,y)] in path order (open)
    returns: dense list[(x,y)] smooth open curve through points
    """
    n = len(points)
    if n < 2:
        return points
    if n == 2:
        # straight line
        return points

    curve = []
    for i in range(n - 1):
        # endpoint padding by clamping indices
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, n - 1)]
        for s in range(samples_per_seg):
            t = s / float(samples_per_seg)
            curve.append(_catmull_rom(p0, p1, p2, p3, t))
    curve.append(points[-1])
    return curve

# --- geometry helpers ---
def sort_points_by_angle(points, center):
    cx, cy = center
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

def _orient(a, b, c):
    # cross product sign of (b-a) x (c-a)
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a, b, c):
    # c collinear with a-b and within bbox
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

def segments_intersect(p1, p2, q1, q2):
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    # general case
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    # collinear cases
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, p2, q2): return True
    if o3 == 0 and _on_segment(q1, q2, p1): return True
    if o4 == 0 and _on_segment(q1, q2, p2): return True
    return False

def would_self_intersect(points, new_pt):
    """
    Check whether adding segment (points[-1] -> new_pt) would intersect
    any earlier segment in the polyline (excluding the adjacent last segment).
    """
    if len(points) < 3:
        return False
    p1 = points[-1]
    p2 = new_pt
    # check against segments (points[i] -> points[i+1]) for i=0..len-3
    for i in range(len(points) - 2):
        q1, q2 = points[i], points[i + 1]
        # skip if sharing an endpoint (shouldn't happen with this loop, but safe)
        if q1 == p1 or q2 == p1:
            continue
        if segments_intersect(p1, p2, q1, q2):
            return True
    return False

def _dist2_point_to_segment(p, a, b):
    # squared distance from p to segment a-b
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = (wx * vx + wy * vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t * vx, ay + t * vy
    dx, dy = px - cx, py - cy
    return dx * dx + dy * dy

def _min_dist2_to_polyline(p, points):
    # distance^2 from p to any earlier segment of polyline
    if len(points) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(points) - 1):
        best = min(best, _dist2_point_to_segment(p, points[i], points[i + 1]))
    return best

### POINT GENERATION FUNCTIONS

def generate_radial_keypoints(
    image_size,            # (W,H)
    shape_size: float,     # intended diameter (can be as large as fits inside 32px margin)
    shape_complexity: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,
    # NEW knobs:
    fill_fraction: float = 0.98,      # how close to the max allowable radius we try to get
    radius_jitter: float = 0.18,      # variation around the target radius (0..~0.4 typical)
):
    """
    Generates keypoints that can expand up to the full available window (inside 'margin'),
    and respects shape_size as a target diameter (clamped to what's possible).

    Key idea: choose center, compute the *true max radius allowed* by margins & center,
    then set target_radius = min(shape_size/2, max_allowed_radius) * fill_fraction
    and jitter around that so the blob can get big.
    """
    if rng is None:
        rng = np.random.default_rng()

    W, H = image_size
    half_target = shape_size / 2.0

    # center is chosen so that a circle of radius half_target would fit *if possible*;
    # if shape_size is huge, we still keep center within margin-only safe window.
    # So: clamp half_target used for center sampling to at most what margins allow globally.
    global_max_half = max(1.0, min((W - 2 * margin) / 2.0, (H - 2 * margin) / 2.0))
    half_for_center = min(half_target, global_max_half)

    x_lo, x_hi = margin + half_for_center, W - margin - half_for_center
    y_lo, y_hi = margin + half_for_center, H - margin - half_for_center
    if x_lo >= x_hi or y_lo >= y_hi:
        # If the image is tight, fall back to any center in the margin window.
        x_lo, x_hi = margin, W - margin
        y_lo, y_hi = margin, H - margin
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("image_size too small for given margin.")

    cx = float(rng.uniform(x_lo, x_hi))
    cy = float(rng.uniform(y_lo, y_hi))
    center = (cx, cy)

    # Per-center maximum radius that keeps all points within the margin window
    max_allowed = min(cx - margin, (W - margin) - cx, cy - margin, (H - margin) - cy)
    max_allowed = max(1.0, float(max_allowed))

    # Shape radius tries to be as large as requested (shape_size/2),
    # but can grow all the way to the available margin window when requested.
    base_r = min(half_target, max_allowed) * float(fill_fraction)

    # Angles: near-uniform with jitter, then sorted (radial order)
    base = np.linspace(0.0, 2.0 * np.pi, shape_complexity, endpoint=False)
    ang_jitter = rng.uniform(-np.pi / shape_complexity, np.pi / shape_complexity, size=shape_complexity)
    angles = np.sort((base + ang_jitter) % (2.0 * np.pi))

    # Radii: centered on base_r with multiplicative jitter
    lo = max(0.05, 1.0 - radius_jitter)
    hi = 1.0 + radius_jitter
    radii = base_r * rng.uniform(lo, hi, size=shape_complexity)
    radii = np.clip(radii, 1.0, max_allowed * 0.999)  # keep safely inside

    pts = []
    for a, r in zip(angles, radii):
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        # keep strictly inside margin window
        x = float(np.clip(x, margin, W - margin))
        y = float(np.clip(y, margin, H - margin))
        pts.append((x, y))

    # Re-sort by angle (in case clamping perturbed order slightly)
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return center, pts

def generate_open_path_points(
    image_size,                  # (W,H)
    num_points: int,             # like "shape_complexity" for curves
    margin: int = 32,
    rng: np.random.Generator | None = None,
    step_range=(18.0, 55.0),     # typical segment length
    max_turn_deg: float = 120.0, # limit sharp turn-backs; lower => smoother
    min_sep: float = 10.0,       # keep points from clumping
    max_tries: int = 5000,       # overall safety cap
):
    """
    Produces an OPEN polyline that:
    - stays within [margin, W-margin] x [margin, H-margin]
    - does NOT self-intersect (no loops)
    - points are not radial; they can be anywhere inside the boundary
    """
    if rng is None:
        rng = np.random.default_rng()

    W, H = image_size
    x_lo, x_hi = margin, W - margin
    y_lo, y_hi = margin, H - margin
    if x_lo >= x_hi or y_lo >= y_hi:
        raise ValueError("image_size too small for given margin.")

    # start anywhere inside the window
    start = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
    points = [start]

    # initial heading random
    heading = float(rng.uniform(0, 2 * math.pi))

    def inside(p):
        return (x_lo <= p[0] <= x_hi) and (y_lo <= p[1] <= y_hi)

    def too_close(p):
        return any((p[0]-q[0])**2 + (p[1]-q[1])**2 < (min_sep**2) for q in points)

    tries = 0
    while len(points) < num_points and tries < max_tries:
        tries += 1

        # pick a turn relative to current heading (biased to continue forward)
        max_turn = math.radians(max_turn_deg)
        dtheta = float(rng.uniform(-max_turn, max_turn))
        # mild bias to small turns (smoother): sample twice and take smaller magnitude
        dtheta2 = float(rng.uniform(-max_turn, max_turn))
        if abs(dtheta2) < abs(dtheta):
            dtheta = dtheta2

        new_heading = heading + dtheta
        step = float(rng.uniform(step_range[0], step_range[1]))
        cand = (points[-1][0] + step * math.cos(new_heading),
                points[-1][1] + step * math.sin(new_heading))

        if not inside(cand):
            continue
        if too_close(cand):
            continue
        if would_self_intersect(points, cand):
            continue

        # accept
        points.append(cand)
        heading = new_heading

    # If we failed to reach num_points, do a restart (common if constraints are tight)
    if len(points) < num_points:
        # one simple retry strategy: call again with looser settings
        return generate_open_path_points(
            image_size=image_size,
            num_points=num_points,
            margin=margin,
            rng=rng,
            step_range=step_range,
            max_turn_deg=max_turn_deg + 20.0,
            min_sep=max(1.0, min_sep * 0.8),
            max_tries=max_tries,
        )

    return points

def generate_unraveled_open_path_points(
    image_size,                  # (W,H)
    num_points: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,

    # “wiggly line” feel:
    step_range=(14.0, 36.0),     # smaller steps = less polygonal wrap
    max_turn_deg: float = 75.0,  # limit sharp turns/backtracking
    forward_min: float = 0.25,   # min fraction of step in global forward direction
    repel_radius: float = 22.0,  # avoid being closer than this to earlier polyline
    end_far_fraction: float = 0.55,  # end must be far from start (relative to max possible)

    # safety:
    max_tries: int = 15000,
):
    """
    Open polyline that looks "unraveled":
    - stays within margin window
    - no self-intersections
    - strongly avoids circling back near start / near its own trail
    - enforces net drift along a global direction (so it can't almost close)

    Returns: list of points in order
    """
    if rng is None:
        rng = np.random.default_rng()

    W, H = image_size
    x_lo, x_hi = margin, W - margin
    y_lo, y_hi = margin, H - margin
    if x_lo >= x_hi or y_lo >= y_hi:
        raise ValueError("image_size too small for given margin.")

    # pick a global forward direction (unit vector)
    phi = float(rng.uniform(0, 2 * math.pi))
    forward = (math.cos(phi), math.sin(phi))

    # start anywhere in window
    start = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
    points = [start]

    # initial heading biased toward forward direction
    heading = phi + float(rng.uniform(-math.radians(25), math.radians(25)))

    def inside(p):
        return (x_lo <= p[0] <= x_hi) and (y_lo <= p[1] <= y_hi)

    repel2 = repel_radius * repel_radius
    max_turn = math.radians(max_turn_deg)

    tries = 0
    while len(points) < num_points and tries < max_tries:
        tries += 1

        # propose a small-ish turn (wiggly but not foldy)
        dtheta = float(rng.uniform(-max_turn, max_turn))
        # bias to smaller turns for smoothness
        dtheta2 = float(rng.uniform(-max_turn, max_turn))
        if abs(dtheta2) < abs(dtheta):
            dtheta = dtheta2

        new_heading = heading + dtheta
        step = float(rng.uniform(step_range[0], step_range[1]))

        cand = (
            points[-1][0] + step * math.cos(new_heading),
            points[-1][1] + step * math.sin(new_heading),
        )

        if not inside(cand):
            continue

        # 1) enforce “unravel” drift: must move forward overall
        dx, dy = cand[0] - points[-1][0], cand[1] - points[-1][1]
        proj = (dx * forward[0] + dy * forward[1])  # signed forward progress
        if proj < forward_min * step:
            continue

        # 2) no self intersections
        if would_self_intersect(points, cand):
            continue

        # 3) repel from earlier trail (prevents almost-closing & hugging itself)
        #    (ignore last segment neighborhood by checking distance to segments anyway)
        if _min_dist2_to_polyline(cand, points) < repel2:
            continue

        # accept
        points.append(cand)
        heading = new_heading

    # If short, retry with slightly relaxed params
    if len(points) < num_points:
        return generate_unraveled_open_path_points(
            image_size=image_size,
            num_points=num_points,
            margin=margin,
            rng=rng,
            step_range=step_range,
            max_turn_deg=min(120.0, max_turn_deg + 15.0),
            forward_min=max(0.05, forward_min * 0.8),
            repel_radius=max(10.0, repel_radius * 0.85),
            end_far_fraction=end_far_fraction,
            max_tries=max_tries,
        )

    # 4) enforce endpoint “not almost closed”: end far from start
    # compute a scale for "far" as a fraction of the window diagonal
    diag = math.hypot((x_hi - x_lo), (y_hi - y_lo))
    min_end_dist = end_far_fraction * diag
    end_dist = math.hypot(points[-1][0] - start[0], points[-1][1] - start[1])
    if end_dist < min_end_dist and tries < max_tries:
        # restart if it still looks too closed
        return generate_unraveled_open_path_points(
            image_size=image_size,
            num_points=num_points,
            margin=margin,
            rng=rng,
            step_range=step_range,
            max_turn_deg=max_turn_deg,
            forward_min=forward_min,
            repel_radius=repel_radius,
            end_far_fraction=max(0.35, end_far_fraction * 0.9),
            max_tries=max_tries,
        )

    return points

def generate_open_curve_fixed_size(
    image_size,
    num_points,
    shape_size,
    margin=32,
    rng=None,
    lateral_frac=0.35,
    mix_beta=0.25,
):
    if rng is None:
        rng = np.random.default_rng()

    W, H = image_size

    # --- endpoints ---
    for _ in range(100):
        center = rng.uniform([margin, margin], [W-margin, H-margin])
        theta = rng.uniform(0, 2*np.pi)
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-u[1], u[0]])

        p0 = center - 0.5 * shape_size * u
        p1 = center + 0.5 * shape_size * u

        if all(margin <= p0) and p0[0] <= W-margin and p0[1] <= H-margin \
           and all(margin <= p1) and p1[0] <= W-margin and p1[1] <= H-margin:
            break
    else:
        raise RuntimeError("Could not place endpoints")

    points = [p0, p1]

    # --- interior sampling ---
    D = shape_size
    alpha = lateral_frac

    interior = []
    for _ in range(num_points - 2):
        t = rng.uniform(0, D)
        s = rng.uniform(-alpha*D, alpha*D)
        p = p0 + t*u + s*v
        interior.append(p)

    # --- ordering ---
    def sort_key(p):
        t = np.dot(p - p0, u)
        s = np.dot(p - p0, v)
        w = np.sin(np.pi * t / D)
        return t + mix_beta * w * s

    interior.sort(key=sort_key)

    return [p0] + interior + [p1]

#### DRAWING FUNCTIONS

def draw_blob(draw: ImageDraw.ImageDraw, keypoints, fill, outline=None, samples_per_seg: int = 25):
    """
    Draw a smooth closed blob through keypoints.
    """
    curve = closed_catmull_rom(keypoints, samples_per_seg=samples_per_seg)
    curve_int = [(int(round(x)), int(round(y))) for (x, y) in curve]
    # polygon fills nicely; outline optional
    draw.polygon(curve_int, fill=fill, outline=outline if outline is not None else fill)

def draw_closed_curve(draw: ImageDraw.ImageDraw, keypoints, center, fill, outline=None, samples_per_seg: int = 25, width: int = 6):
    """
    Draw a smooth closed curve through keypoints.
    """
    curve = closed_catmull_rom(keypoints, samples_per_seg=samples_per_seg)
    curve_int = [(int(round(x)), int(round(y))) for (x, y) in curve]
    curve_int = sort_points_by_angle(curve_int, center)
    # polygon fills nicely; outline optional
    draw.line(curve_int, fill=fill, width=width)

def draw_open_curve(draw: ImageDraw.ImageDraw, points, color, width: int = 6, samples_per_seg: int = 25):
    curve = open_catmull_rom(points, samples_per_seg=samples_per_seg)
    draw.line(curve, fill=color, width=width, joint="curve")


#### ADD DRAWING TO IMAGE WRAPPER FUNCTIONS

def add_random_blob_to_image(
    image,
    draw: ImageDraw.ImageDraw,
    shape_color,
    shape_size: float,
    shape_complexity: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,
):
    (W, H) = image.size
    center, keypoints = generate_radial_keypoints(
        image_size=(W, H),
        shape_size=shape_size,
        shape_complexity=shape_complexity,
        margin=margin,
        rng=rng,
        fill_fraction=0.99,     # push close to boundary
        radius_jitter=0.22,     # still “blobby”
    )
    draw_blob(draw, keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
    return center, keypoints

def add_random_closed_curve_to_image(
    image,
    draw: ImageDraw.ImageDraw,
    shape_color,
    shape_size: float,
    shape_complexity: int,
    margin: int = 32,
    width: int = 6,
    rng: np.random.Generator | None = None,
):
    (W, H) = image.size
    center, keypoints = generate_radial_keypoints(
        image_size=(W, H),
        shape_size=shape_size,
        shape_complexity=shape_complexity,
        margin=margin,
        rng=rng,
        fill_fraction=0.99,     # push close to boundary
        radius_jitter=0.22,     # still “blobby”
    )
    draw_closed_curve(draw, keypoints, center, fill=shape_color, outline=shape_color, samples_per_seg=30, width=width)
    return center, keypoints

def add_random_unraveled_curve_to_image(
    image,
    draw,
    curve_color,
    shape_complexity: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,
    width: int = 6,
):
    pts = generate_unraveled_open_path_points(
        image_size=image.size,
        num_points=shape_complexity,
        margin=margin,
        rng=rng,
        step_range=(14.0, 36.0),
        max_turn_deg=75.0,
        forward_min=0.25,
        repel_radius=22.0,
        end_far_fraction=0.55,
    )
    draw_open_curve(draw, pts, color=curve_color, width=width, samples_per_seg=25)
    return pts

def draw_shape(
    shape_type: str,
    image,
    draw,
    shape_color,
    shape_complexity: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,
    width: int = 6,
    shape_size: float = 100.0,
):
    if rng is None:
        rng = np.random.default_rng()
    
    if shape_type == "circle" or shape_type == "square":
        x_range = (32 + shape_size/2, image.size[0] - 32 - shape_size/2)
        y_range = (32 + shape_size/2, image.size[1] - 32 - shape_size/2)

        x_pos = rng.randint(x_range[0], x_range[1])
        y_pos = rng.randint(y_range[0], y_range[1])

        if shape_type == "circle":
            draw.ellipse(
                (x_pos - shape_size/2, y_pos - shape_size/2, x_pos + shape_size/2, y_pos + shape_size/2),
                fill=shape_color,
                outline=shape_color,
            )
        else:
            draw.rectangle(
                (x_pos - shape_size/2, y_pos - shape_size/2, x_pos + shape_size/2, y_pos + shape_size/2),
                fill=shape_color,
                outline=shape_color,
            )
    elif shape_type == "oval" or shape_type == "rectangle":
        long_axis = rng.choice(["x", "y"])
        short_size = rng.randint(int(shape_size//3), int(shape_size//2))

        if long_axis == "x":
            x_range = (32 + shape_size/2, image.size[0] - 32 - shape_size/2)
            y_range = (32 + short_size/2, image.size[1] - 32 - short_size/2)
            x_length = shape_size
            y_length = short_size
        else:
            y_range = (32 + shape_size/2, image.size[1] - 32 - shape_size/2)
            x_range = (32 + short_size/2, image.size[0] - 32 - short_size/2)
            x_length = short_size
            y_length = shape_size

        x_pos = rng.randint(x_range[0], x_range[1])
        y_pos = rng.randint(y_range[0], y_range[1])

        if shape_type == "oval":
            draw.ellipse(
                (x_pos - x_length/2, y_pos - y_length/2, x_pos + x_length/2, y_pos + y_length/2),
                fill=shape_color,
                outline=shape_color
            ) 
        else:
            draw.rectangle(
                (x_pos - x_length/2, y_pos - y_length/2, x_pos + x_length/2, y_pos + y_length/2),
                fill=shape_color,
                outline=shape_color
            ) 
    elif shape_type == "curve":
        add_random_unraveled_curve_to_image(
            image,
            draw,
            curve_color=shape_color,
            shape_complexity=shape_complexity,
            margin=margin,
            rng=rng,
            width=width,
        )
    elif shape_type == "blob":
        add_random_blob_to_image(
            image,
            draw,
            shape_color,
            shape_size=shape_size,
            shape_complexity=shape_complexity,
            margin=margin,
            rng=rng,
        )
    elif shape_type == "closedcurve":
        add_random_closed_curve_to_image(
            image,
            draw,
            shape_color,
            shape_size=shape_size,
            shape_complexity=shape_complexity,
            margin=margin,
            width=width,
            rng=rng,
        )
    else:
        raise ValueError(f"Invalid shape type: {shape_type}")