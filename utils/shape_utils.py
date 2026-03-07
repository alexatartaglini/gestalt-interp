import math
import numpy as np
from PIL import ImageDraw
from scipy.ndimage import gaussian_filter1d

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

def satisfies_min_angle(points, max_turn_deg):
    """
    Enforces that the direction change (turn angle) at each interior point
    does not exceed max_turn_deg.
    """
    if len(points) < 3:
        return True

    max_turn = math.radians(max_turn_deg)

    for i in range(1, len(points) - 1):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        v_in  = (x1 - x0, y1 - y0)
        v_out = (x2 - x1, y2 - y1)

        n1 = math.hypot(*v_in)
        n2 = math.hypot(*v_out)
        if n1 == 0 or n2 == 0:
            return False

        cos_theta = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (n1 * n2)
        cos_theta = max(-1.0, min(1.0, cos_theta))

        theta = math.acos(cos_theta)
        turn_angle = math.pi - theta

        if turn_angle > max_turn:
            return False

    return True

def rotate_to_min_turn(v_in, v_out, min_angle):
    """
    Adjust v_out so that min(angle, 2π-angle) == min_angle.
    Returns new v_out (same length).
    """
    # normalize
    n_out = np.linalg.norm(v_out)
    if n_out == 0:
        return v_out

    v_in_n = v_in / np.linalg.norm(v_in)
    v_out_n = v_out / n_out

    cosθ = np.dot(v_in_n, v_out_n)
    cosθ = np.clip(cosθ, -1.0, 1.0)

    θ = math.acos(cosθ)
    turn = min(θ, 2 * math.pi - θ)

    if turn >= min_angle:
        return v_out  # already valid

    # determine rotation direction
    cross = v_in_n[0] * v_out_n[1] - v_in_n[1] * v_out_n[0]
    sign = np.sign(cross) if cross != 0 else 1.0

    # rotate v_out_n to exactly min_angle
    target_angle = min_angle * sign
    c, s = math.cos(target_angle), math.sin(target_angle)

    R = np.array([[c, -s], [s, c]])
    v_new = R @ v_in_n

    return v_new * n_out

def relax_turn_angles(points, min_angle_deg, alpha=0.7):
    """
    One Gauss–Seidel relaxation pass.
    alpha < 1 damps updates for stability.
    """
    min_angle = math.radians(min_angle_deg)
    pts = [np.array(p, dtype=float) for p in points]

    for i in range(1, len(pts) - 1):
        v_in = pts[i - 1] - pts[i]
        v_out = pts[i + 1] - pts[i]

        v_new = rotate_to_min_turn(v_in, v_out, min_angle)
        pts[i + 1] = pts[i] + alpha * v_new + (1 - alpha) * v_out

    return [tuple(p) for p in pts]

def enforce_min_turn_angles(
    points,
    min_angle_deg,
    max_iters=15,
    tol=1e-3,
):
    pts = points
    for _ in range(max_iters):
        if satisfies_min_angle(pts, min_angle_deg):
            return pts
        new_pts = relax_turn_angles(pts, min_angle_deg)
        shift = max(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(pts, new_pts)
        )
        pts = new_pts
        if shift < tol:
            break
    return pts

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

def _rng_integers(rng, low: int, high: int) -> int:
    """
    Compatibility wrapper for both numpy Generator and RandomState.
    `high` is exclusive, matching Generator.integers / randint semantics.
    """
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))

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
    center_bounds: tuple[float, float] | None = None,
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
    if center_bounds is None:
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

    else:
        (x_lo, x_hi), (y_lo, y_hi) = center_bounds
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("center_bounds too small for given margin.")

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

def generate_blob_keypoints(
    image_size,
    shape_size: float,
    shape_complexity: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,
    fill_fraction: float = 0.99,
    center_bounds: tuple[float, float] | None = None,
):
    """
    Generates smooth closed blobs with broad, deep bumps.

    The contour is defined by a relatively small set of anchor points.
    A few large-scale bulges and dents are applied across neighboring
    anchors, then the spline in `draw_blob` smooths that coarse contour
    into a blob with pronounced lobes instead of shallow perturbations.
    """
    if rng is None:
        rng = np.random.default_rng()

    W, H = image_size
    half_target = shape_size / 2.0
    complexity = max(3, int(shape_complexity))
    complexity_scale = min(1.0, (complexity - 3) / 7.0)

    if center_bounds is None:
        global_max_half = max(1.0, min((W - 2 * margin) / 2.0, (H - 2 * margin) / 2.0))
        half_for_center = min(half_target, global_max_half)

        x_lo, x_hi = margin + half_for_center, W - margin - half_for_center
        y_lo, y_hi = margin + half_for_center, H - margin - half_for_center
        if x_lo >= x_hi or y_lo >= y_hi:
            x_lo, x_hi = margin, W - margin
            y_lo, y_hi = margin, H - margin
            if x_lo >= x_hi or y_lo >= y_hi:
                raise ValueError("image_size too small for given margin.")
    else:
        (x_lo, x_hi), (y_lo, y_hi) = center_bounds
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("center_bounds too small for given margin.")

    cx = float(rng.uniform(x_lo, x_hi))
    cy = float(rng.uniform(y_lo, y_hi))
    center = (cx, cy)

    max_allowed = min(cx - margin, (W - margin) - cx, cy - margin, (H - margin) - cy)
    max_allowed = max(1.0, float(max_allowed))
    base_r = min(half_target, max_allowed) * float(fill_fraction)

    min_points = max(5, complexity + 1)
    max_points = max(min_points + 1, complexity + 4)
    num_points = _rng_integers(rng, min_points, max_points + 1)

    angle_step = (2.0 * np.pi) / num_points
    base_angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    angle_jitter = rng.uniform(
        -np.interp(complexity_scale, [0.0, 1.0], [0.18, 0.32]) * angle_step,
        np.interp(complexity_scale, [0.0, 1.0], [0.18, 0.32]) * angle_step,
        size=num_points,
    )
    angles = np.sort((base_angles + angle_jitter) % (2.0 * np.pi))

    min_radius_scale = np.interp(complexity_scale, [0.0, 1.0], [0.7, 0.28])
    max_radius_scale = np.interp(complexity_scale, [0.0, 1.0], [1.2, 1.55])
    radii = base_r * rng.uniform(min_radius_scale, max_radius_scale, size=num_points)

    # Apply broad dents/bulges across adjacent anchors so the silhouette
    # gets deep lobes rather than tiny surface texture.
    feature_count = min(num_points, _rng_integers(rng, 2, 3 + max(1, complexity // 3)))
    for _ in range(feature_count):
        center_idx = _rng_integers(rng, 0, num_points)
        half_width = _rng_integers(rng, 1, min(3, max(2, num_points // 3)))
        if rng.uniform() < 0.5:
            strength = rng.uniform(
                np.interp(complexity_scale, [0.0, 1.0], [0.18, 0.28]),
                np.interp(complexity_scale, [0.0, 1.0], [0.32, 0.55]),
            )
        else:
            strength = -rng.uniform(
                np.interp(complexity_scale, [0.0, 1.0], [0.2, 0.3]),
                np.interp(complexity_scale, [0.0, 1.0], [0.38, 0.62]),
            )

        for offset in range(-half_width, half_width + 1):
            idx = (center_idx + offset) % num_points
            distance = abs(offset) / max(1, half_width)
            weight = 0.5 * (1.0 + math.cos(np.pi * distance))
            radii[idx] *= (1.0 + strength * weight)

    # Add a coarse anisotropy term so many samples become obviously
    # elongated / lopsided rather than staying near-circular.
    stretch_axis = float(rng.uniform(0.0, 2.0 * np.pi))
    stretch_strength = rng.uniform(0.08, np.interp(complexity_scale, [0.0, 1.0], [0.18, 0.35]))
    radii *= 1.0 + stretch_strength * np.cos(angles - stretch_axis)

    radii = np.clip(radii, base_r * 0.16, max_allowed * 0.999)

    pts = []
    for a, r in zip(angles, radii):
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        x = float(np.clip(x, margin, W - margin))
        y = float(np.clip(y, margin, H - margin))
        pts.append((x, y))

    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return center, pts

def generate_close_blob_keypoints(
    image_size,
    shape_size: float,
    shape_complexity: int,
    margin: int = 0,
    rng: np.random.Generator | None = None,
    fill_fraction: float = 0.995,
):
    """
    Generates a large, bumpy closed contour for close-blob scenes.
    Base support can be circular or oval, then high-frequency radial bumps are
    added so the contour is noticeably rougher than standard blobs.
    """
    if rng is None:
        rng = np.random.default_rng()

    W, H = image_size
    base_radius = max(4.0, float(shape_size) / 2.0)
    if (W - 2 * margin) > 2 * base_radius and (H - 2 * margin) > 2 * base_radius:
        cx = float(rng.uniform(margin + base_radius, W - margin - base_radius))
        cy = float(rng.uniform(margin + base_radius, H - margin - base_radius))
    else:
        cx = float(rng.uniform(margin, W - margin))
        cy = float(rng.uniform(margin, H - margin))
    center = (cx, cy)

    max_rx = max(1.0, min(cx - margin, (W - margin) - cx))
    max_ry = max(1.0, min(cy - margin, (H - margin) - cy))

    # Keep close blobs mostly round to avoid "egg-like" silhouettes.
    # Only allow mild anisotropy.
    if rng.uniform() < 0.8:
        base_rx = base_radius
        base_ry = base_radius
    else:
        aspect = float(rng.uniform(0.9, 0.98))
        if rng.uniform() < 0.5:
            base_rx = base_radius
            base_ry = base_radius * aspect
        else:
            base_rx = base_radius * aspect
            base_ry = base_radius

    base_rx = min(base_rx * fill_fraction, max_rx * 0.995)
    base_ry = min(base_ry * fill_fraction, max_ry * 0.995)
    base_rx = max(2.0, base_rx)
    base_ry = max(2.0, base_ry)

    complexity = max(10, int(shape_complexity))
    num_points = max(36, complexity * 5)
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    # Low-frequency random harmonic profile:
    # more globally irregular and less "spiky symmetric".
    radial_mod = np.ones(num_points, dtype=float)
    for k in range(2, 8):
        coeff = float(rng.normal(0.0, 0.18 / (k ** 0.65)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        radial_mod += coeff * np.sin(k * angles + phase)

    # Broad asymmetric lobes/dents (organic "splat" look).
    n_lobes = _rng_integers(rng, 4, 10)
    for _ in range(n_lobes):
        c = float(rng.uniform(0.0, 2.0 * np.pi))
        w = float(rng.uniform(0.22, 0.55))
        sgn = -1.0 if rng.uniform() < 0.4 else 1.0
        a = sgn * float(rng.uniform(0.05, 0.22))
        d = np.angle(np.exp(1j * (angles - c)))  # wrapped angle difference
        radial_mod += a * np.exp(-(d * d) / (2.0 * w * w))

    # Tiny residual texture only; keep silhouette smooth.
    fine_noise = rng.normal(0.0, 1.0, size=num_points)
    fine_noise = gaussian_filter1d(fine_noise, sigma=max(1.0, num_points / 36.0), mode="wrap")
    radial_mod += float(rng.uniform(0.01, 0.04)) * fine_noise / (np.max(np.abs(fine_noise)) + 1e-8)

    radial_mod = gaussian_filter1d(radial_mod, sigma=max(0.8, num_points / 56.0), mode="wrap")
    radial_mod = np.clip(radial_mod, 0.68, 1.65)

    pts = []
    for a, m in zip(angles, radial_mod):
        x = cx + base_rx * m * math.cos(a)
        y = cy + base_ry * m * math.sin(a)
        x = float(np.clip(x, margin, W - margin))
        y = float(np.clip(y, margin, H - margin))
        pts.append((x, y))

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

    # enforce minimum angle between segments
    min_angle_deg: float = 100.0,

    # safety:
    max_tries: int = 15000,

    center_bounds: tuple[float, float] | None = None,
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

    if center_bounds is None:
        x_lo, x_hi = margin, W - margin
        y_lo, y_hi = margin, H - margin
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("image_size too small for given margin.")
    else:
        (x_lo, x_hi), (y_lo, y_hi) = center_bounds
        if x_lo >= x_hi or y_lo >= y_hi:
            raise ValueError("center_bounds too small for given margin.")

    complexity = max(2, int(num_points))
    window_diag = math.hypot((x_hi - x_lo), (y_hi - y_lo))

    def _inside(p):
        return (x_lo <= p[0] <= x_hi) and (y_lo <= p[1] <= y_hi)

    def _fallback_points():
        # Guaranteed, bounded fallback: monotone-ish open polyline.
        phi = float(rng.uniform(0, 2 * math.pi))
        heading = phi
        step_base = max(2.0, 0.45 * (step_range[0] + step_range[1]))
        start = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
        out = [start]
        for _ in range(complexity - 1):
            placed = False
            for _ in range(60):
                step = float(rng.uniform(0.7 * step_base, 1.3 * step_base))
                heading_try = heading + float(rng.uniform(-math.radians(28), math.radians(28)))
                cand = (
                    out[-1][0] + step * math.cos(heading_try),
                    out[-1][1] + step * math.sin(heading_try),
                )
                if _inside(cand):
                    out.append(cand)
                    heading = heading_try
                    placed = True
                    break
            if not placed:
                out.append((
                    float(np.clip(out[-1][0], x_lo, x_hi)),
                    float(np.clip(out[-1][1], y_lo, y_hi)),
                ))
        return out

    # Replace recursive retries with bounded restarts.
    max_restarts = 28
    tries_per_restart = max(200, int(max_tries // max_restarts))
    step_lo = float(max(1.0, min(step_range)))
    step_hi = float(max(step_lo + 1e-6, max(step_range)))
    turn_deg = float(max_turn_deg)
    fwd_min = float(forward_min)
    repel_r = float(repel_radius)
    end_far = float(end_far_fraction)
    min_ang = float(min_angle_deg)

    for _ in range(max_restarts):
        # pick a global forward direction (unit vector)
        phi = float(rng.uniform(0, 2 * math.pi))
        forward = (math.cos(phi), math.sin(phi))
        start = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
        points = [start]
        heading = phi + float(rng.uniform(-math.radians(25), math.radians(25)))

        repel2 = repel_r * repel_r
        max_turn = math.radians(turn_deg)

        tries = 0
        while len(points) < complexity and tries < tries_per_restart:
            tries += 1
            dtheta = float(rng.uniform(-max_turn, max_turn))
            dtheta2 = float(rng.uniform(-max_turn, max_turn))
            if abs(dtheta2) < abs(dtheta):
                dtheta = dtheta2

            new_heading = heading + dtheta
            step = float(rng.uniform(step_lo, step_hi))
            cand = (
                points[-1][0] + step * math.cos(new_heading),
                points[-1][1] + step * math.sin(new_heading),
            )

            if not _inside(cand):
                continue

            dx, dy = cand[0] - points[-1][0], cand[1] - points[-1][1]
            proj = (dx * forward[0] + dy * forward[1])
            if proj < fwd_min * step:
                continue
            if would_self_intersect(points, cand):
                continue
            if _min_dist2_to_polyline(cand, points) < repel2:
                continue

            points.append(cand)
            heading = new_heading

        if len(points) == complexity:
            # Endpoint should not look almost closed.
            min_end_dist = max(0.0, end_far) * window_diag
            end_dist = math.hypot(points[-1][0] - start[0], points[-1][1] - start[1])
            if end_dist < min_end_dist:
                # Relax and retry.
                end_far = max(0.2, end_far * 0.9)
                turn_deg = min(130.0, turn_deg + 4.0)
                fwd_min = max(0.02, fwd_min * 0.9)
                continue

            points = enforce_min_turn_angles(points, min_ang)
            if satisfies_min_angle(points, min_ang):
                return points

        # Progressive relaxation for next restart
        turn_deg = min(130.0, turn_deg + 7.0)
        fwd_min = max(0.02, fwd_min * 0.87)
        repel_r = max(2.0, repel_r * 0.85)
        end_far = max(0.2, end_far * 0.9)
        min_ang = max(35.0, min_ang * 0.92)
        step_lo = max(1.0, step_lo * 0.93)
        step_hi = max(step_lo + 1.0, step_hi * 0.95)

    return _fallback_points()

def generate_backbone(image_size, num_points, shape_size, margin, rng, center_bounds=None):
    W, H = image_size
    if center_bounds is None:
        x_lo, x_hi = margin, W - margin
        y_lo, y_hi = margin, H - margin
    else:
        (x_lo, x_hi), (y_lo, y_hi) = center_bounds

    # random forward direction
    phi = rng.uniform(0, 2 * math.pi)
    d = np.array([math.cos(phi), math.sin(phi)])

    # choose start so end stays inside
    center = np.array([
        rng.uniform(x_lo + shape_size, x_hi - shape_size),
        rng.uniform(y_lo + shape_size, y_hi - shape_size),
    ])

    start = center - 0.5 * shape_size * d
    end   = center + 0.5 * shape_size * d

    t = np.linspace(0, 1, num_points)
    backbone = (1 - t)[:, None] * start + t[:, None] * end
    return backbone

def gaussian_smooth_1d(x, sigma):
    radius = int(3 * sigma)
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")

def add_lateral_noise(backbone, max_amp, smooth_sigma, rng):
    pts = backbone.copy()

    # tangent directions
    tangents = np.gradient(pts, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # normals
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    noise = rng.normal(0, 1, size=len(pts))
    noise = gaussian_smooth_1d(noise, smooth_sigma)
    noise /= np.max(np.abs(noise)) + 1e-8

    pts += normals * noise[:, None] * max_amp
    return pts

def clamp_points(pts, image_size, margin):
    W, H = image_size
    pts[:, 0] = np.clip(pts[:, 0], margin, W - margin)
    pts[:, 1] = np.clip(pts[:, 1], margin, H - margin)
    return pts

def generate_open_curve_points(
    image_size,
    num_points,
    shape_size=120.0,
    wiggle_amp=20.0,
    wiggle_smooth=6.0,
    margin=32,
    rng=None,
    center_bounds=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # Curve complexity is controlled by num_points, and overall curve length
    # is controlled by shape_size via the per-segment step range.
    complexity = max(3, int(num_points))
    target_length = max(8.0, float(shape_size))
    avg_step = target_length / max(1, complexity - 1)

    # Keep steps in a stable range while preserving the requested length scale.
    step_lo = max(3.0, 0.7 * avg_step)
    step_hi = max(step_lo + 1.0, 1.3 * avg_step)
    if center_bounds is None:
        span_x = image_size[0] - 2 * margin
        span_y = image_size[1] - 2 * margin
    else:
        (x_lo, x_hi), (y_lo, y_hi) = center_bounds
        span_x = x_hi - x_lo
        span_y = y_hi - y_lo
    max_step = max(3.5, 0.55 * min(span_x, span_y))
    step_hi = min(step_hi, max_step)
    step_lo = min(step_lo, max(3.0, 0.8 * step_hi))

    pts = generate_unraveled_open_path_points(
        image_size=image_size,
        num_points=complexity,
        margin=margin,
        rng=rng,
        step_range=(step_lo, step_hi),
        # Larger wiggle_amp allows bendier paths, but keep them unraveled.
        max_turn_deg=float(np.clip(35.0 + 0.4 * float(wiggle_amp), 35.0, 85.0)),
        forward_min=0.2,
        repel_radius=max(8.0, 0.25 * avg_step),
        end_far_fraction=0.5,
        min_angle_deg=max(70.0, 130.0 - 2.5 * float(wiggle_smooth)),
        center_bounds=center_bounds,
    )
    return [tuple(p) for p in pts]

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
    center_bounds: tuple[float, float] | None = None,
):
    (W, H) = image.size
    center, keypoints = generate_blob_keypoints(
        image_size=(W, H),
        shape_size=shape_size,
        shape_complexity=shape_complexity,
        margin=margin,
        rng=rng,
        fill_fraction=0.99,
        center_bounds=center_bounds,
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
    center_bounds: tuple[float, float] | None = None,
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
        center_bounds=center_bounds,
    )
    draw_closed_curve(draw, keypoints, center, fill=shape_color, outline=shape_color, samples_per_seg=30, width=width)
    return center, keypoints

def add_random_unraveled_curve_to_image(
    image,
    draw,
    curve_color,
    shape_complexity: int,
    shape_size: int,
    margin: int = 32,
    rng: np.random.Generator | None = None,
    width: int = 6,
    center_bounds: tuple[float, float] | None = None,
):
    pts = generate_open_curve_points(
        image_size=image.size,
        num_points=shape_complexity,
        shape_size=shape_size,
        wiggle_amp=0.4 * shape_size,
        wiggle_smooth=0.15 * shape_complexity,
        margin=margin,
        rng=rng,
        center_bounds=center_bounds,
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
    shape_size: float = 20.0,
    center_bounds: tuple[float, float] | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    W, H = image.size
    
    if shape_type == "circle" or shape_type == "square":
        if center_bounds is None:
            x_pos = rng.randint(32 + shape_size/2, W - 32 - shape_size/2)
            y_pos = rng.randint(32 + shape_size/2, H - 32 - shape_size/2)
        else:
            (xmin, xmax), (ymin, ymax) = center_bounds
            x_pos = rng.randint(xmin, xmax)
            y_pos = rng.randint(ymin, ymax)

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
            if center_bounds is None:
                x_range = (32 + shape_size/2, W - 32 - shape_size/2)
                y_range = (32 + short_size/2, H - 32 - short_size/2)
            else:
                (xmin, xmax), (ymin, ymax) = center_bounds
                x_range = (xmin, xmax)
                y_range = (ymin, ymax)
            x_length = shape_size
            y_length = short_size
        else:
            if center_bounds is None:
                y_range = (32 + shape_size/2, H - 32 - shape_size/2)
                x_range = (32 + short_size/2, W - 32 - short_size/2)
            else:
                (xmin, xmax), (ymin, ymax) = center_bounds
                y_range = (ymin, ymax)
                x_range = (xmin, xmax)
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
            shape_size=shape_size,
            margin=margin,
            rng=rng,
            width=width,
            center_bounds=center_bounds,
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
            center_bounds=center_bounds,
        )
    elif shape_type == "close_blobs":
        center, keypoints = generate_close_blob_keypoints(
            image_size=(W, H),
            shape_size=shape_size,
            shape_complexity=shape_complexity,
            margin=0,
            rng=rng,
            fill_fraction=0.998,
        )
        draw_blob(draw, keypoints, fill=shape_color, outline=shape_color, samples_per_seg=30)
        return center, keypoints
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
            center_bounds=center_bounds,
        )
    else:
        raise ValueError(f"Invalid shape type: {shape_type}")
