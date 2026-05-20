"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
import numpy as np

def intersection_with_rectangle(cx, cy, w, h, px, py):
    # Half sizes
    hw = w / 2.0
    hh = h / 2.0

    # Direction vector from center to point
    dx = px - cx
    dy = py - cy

    # Avoid division by zero
    if dx == 0:
        t_y = (hh if dy > 0 else -hh) / dy
        return cx, cy + t_y * dy

    if dy == 0:
        t_x = (hw if dx > 0 else -hw) / dx
        return cx + t_x * dx, cy

    # Find intersection scale for x boundaries
    tx1 = ( hw) / dx
    tx2 = (-hw) / dx

    # Find intersection scale for y boundaries
    ty1 = ( hh) / dy
    ty2 = (-hh) / dy

    # Only positive t values (in the direction of the ray)
    ts = [t for t in [tx1, tx2, ty1, ty2] if t > 0]

    # Take the smallest positive t (first boundary hit)
    t = min(ts)

    # Intersection point
    return cx + t*dx, cy + t*dy


# == margin ==
def calculate_margin_rect(s, x_y_w_h, negativeInside=True):
  """Calculates the margin to a rectangular box in the x-y state space.

    Args:
        s (np.ndarray): the state of the agent. It requires that s[0] is the
            x position and s[1] is the y position.
        x_y_w_h (tuple of floats): (center_x, center_y, width, height).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
  x, y, w, h = x_y_w_h

  x_r, y_r = intersection_with_rectangle(x, y, w, h, s[0], s[1])

  if s[0] >= x - w/2 and s[0] <= x + w/2 and s[1] >= y - h/2 and s[1] <= y + h/2:
    margin = -np.sqrt((x_r - x)**2 + (y_r - y)**2) - np.sqrt((s[0] - x)**2 + (s[1] - y)**2)
  else:
    margin = -np.sqrt((x_r - x)**2 + (y_r - y)**2) + np.sqrt((s[0] - x)**2 + (s[1] - y)**2)
#   delta_x = np.abs(s[0] - x)
#   delta_y = np.abs(s[1] - y)
#   margin = max(delta_y - h/2, delta_x - w/2)

  if negativeInside:
    return margin
  else:
    return -margin


def calculate_margin_circle(s, c_r, negativeInside=True):
  """Calculates the margin to a circle in the x-y state space.

    Args:
        s (np.ndarray): the state of the agent. It requires that s[0] is the
            x position and s[1] is the y position.
        c_r (tuple of np.ndarray and float)): (center, radius).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the circle. Defaults to True.

    Returns:
        float: margin.
    """
  center, radius = c_r
  dist_to_center = np.linalg.norm(s[:2] - center)
  margin = dist_to_center - radius

  if negativeInside:
    return margin
  else:
    return -margin

def intersection_with_rectangle_batch(cx, cy, w, h, px, py):
    """Ray from rectangle center through each point, intersected with boundary. Batched."""
    dx = px - cx  # (N,)
    dy = py - cy  # (N,)

    # t_x, t_y: how far along the ray before hitting each pair of walls
    with np.errstate(divide='ignore', invalid='ignore'):
        t_x = np.where(np.abs(dx) > 1e-10, (w / 2) / np.abs(dx), np.inf)
        t_y = np.where(np.abs(dy) > 1e-10, (h / 2) / np.abs(dy), np.inf)

    t = np.minimum(t_x, t_y)  # first wall hit
    return cx + t * dx, cy + t * dy  # (N,), (N,)


def calculate_margin_rect_batch(positions, x_y_w_h, negativeInside=True):
    """Batched version of calculate_margin_rect. positions: (N, 2)"""
    cx, cy, w, h = x_y_w_h
    px, py = positions[:, 0], positions[:, 1]

    x_r, y_r = intersection_with_rectangle_batch(cx, cy, w, h, px, py)

    dist_boundary = np.sqrt((x_r - cx)**2 + (y_r - cy)**2)  # (N,)
    dist_point    = np.sqrt((px  - cx)**2 + (py  - cy)**2)  # (N,)

    inside = (px >= cx - w/2) & (px <= cx + w/2) & (py >= cy - h/2) & (py <= cy + h/2)
    margin = np.where(inside, -dist_boundary - dist_point,
                               -dist_boundary + dist_point)  # (N,)

    return margin if negativeInside else -margin


def calculate_margin_circle_batch(positions, c_r, negativeInside=True):
    """Batched version of calculate_margin_circle. positions: (N, 2)"""
    center, radius = c_r
    dist = np.linalg.norm(positions - center, axis=1)  # (N,)
    margin = dist - radius
    return margin if negativeInside else -margin

# == Plotting ==
def plot_arc(
    center, r, thetaParam, ax, c='b', lw=1.5, orientation=0., zorder=0
):
  """Plots an arc given a center, a radius and the (theta_init, theta_final).

  Args:
      center (np.ndarray): the center of the arc.
      r (float): the radius of the arc.
      thetaParam (np.ndarray): the initial angle and the final angle.
      ax (matplotlib.axes.Axes): ax to plot.
      c (str, optional): color of the arc. Defaults to 'b'.
      lw (float, optional): linewidth of the arc. Defaults to 1.5.
      orientation (float, optional): rotate points in the xy-plane
          counterclockwise through orientation with respect to the x axis.
          Defaults to 0.
      zorder (int, optional): graph layers order. Defaults to 0.
  """
  x, y = center
  thetaInit, thetaFinal = thetaParam

  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(thetaInit + orientation, thetaFinal + orientation, 100)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)

  ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_circle(
    center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False,
    zorder=0
):
  """Plots a circle given a center and a radius.

  Args:
      enter (np.ndarray): the center of the arc.
      r (float): the radius of the arc.
      ax (matplotlib.axes.Axes): ax to plot.
      c (str, optional): color of the circle. Defaults to 'b'.
      lw (float, optional): linewidth of the circle. Defaults to 1.5.
      ls (str, optional): linestyle of the circle. Defaults to '-'.
      orientation (int, optional): rotate points in the xy-plane
          counterclockwise through orientation with respect to the x axis.
          Defaults to 0.
      scatter (bool, optional): show the centerif True. Defaults to False.
      zorder (int, optional): graph layers order. Defaults to 0.
  """
  x, y = center
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(0, 2 * np.pi, 200)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)
  ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
  if scatter:
    ax.scatter(xtilde + r, ytilde, c=c, s=80)
    ax.scatter(xtilde - r, ytilde, c=c, s=80)
    print(xtilde + r, ytilde, xtilde - r, ytilde)


def rotatePoint(state, orientation):
  """Rotates the point counter-clockwise by a given angle.

  Args:
      state (np.ndarray): the state of the agent. It requires that s[0] is the
          x position and s[1] is the y position.
      orientation (int, optional): counter-clockwise angle.

  Returns:
      np.ndarray: rotated state.
  """
  x, y, theta = state
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)
  thetatilde = theta + orientation

  return np.array([xtilde, ytilde, thetatilde])
