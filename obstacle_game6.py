import pygame
import numpy as np
import cvxpy as cp
import time

DT = 0.1
N = 50
DAMPING = 0.98


def particle_dynamics(x, u):
    px, py, vx, vy = x
    ax, ay = u
    px_next = px + vx * DT
    py_next = py + vy * DT
    vx_next = vx * DAMPING + ax * DT
    vy_next = vy * DAMPING + ay * DT

    ENV_LEFT = 0 + BOUND_MARGIN
    ENV_TOP = 0 + BOUND_MARGIN
    ENV_RIGHT = WIDTH - BOUND_MARGIN
    ENV_BOTTOM = HEIGHT - BOUND_MARGIN
    bounce_damping = 0.5

    if px_next < ENV_LEFT:
        px_next = ENV_LEFT
        vx_next *= -bounce_damping
    elif px_next > ENV_RIGHT:
        px_next = ENV_RIGHT
        vx_next *= -bounce_damping

    if py_next < ENV_TOP:
        py_next = ENV_TOP
        vy_next *= -bounce_damping
    elif py_next > ENV_BOTTOM:
        py_next = ENV_BOTTOM
        vy_next *= -bounce_damping

    return np.array([px_next, py_next, vx_next, vy_next])


def linearize_particle_dynamics():
    A = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, DAMPING, 0], [0, 0, 0, DAMPING]])
    B = np.array([[0, 0], [0, 0], [DT, 0], [0, DT]])
    return A, B


def solve_feedback_control(r_k1, v_k, start_state, rho, obstacles=None):
    n_states = 4
    n_controls = 2
    A, B = linearize_particle_dynamics()
    Q_base = np.eye(n_states) * rho
    R = np.eye(n_controls) * 0.1
    r_target = r_k1 - v_k

    P = np.zeros((N + 1, n_states, n_states))
    p = np.zeros((N + 1, n_states))
    P[N] = Q_base
    p[N] = -Q_base @ r_target[N]

    K_feedback = np.zeros((N, n_controls, n_states))
    k_feedforward = np.zeros((N, n_controls))

    for t in reversed(range(N)):
        Q_t = Q_base.copy()
        R_t = R
        q_t = -Q_base @ r_target[t]

        Q_x = q_t + A.T @ p[t + 1]
        Q_u = B.T @ p[t + 1]
        Q_xx = Q_t + A.T @ P[t + 1] @ A
        Q_ux = B.T @ P[t + 1] @ A
        Q_uu = R_t + B.T @ P[t + 1] @ B

        Q_uu_reg = Q_uu + np.eye(n_controls) * 1e-6
        try:
            Q_uu_inv = np.linalg.inv(Q_uu_reg)
        except np.linalg.LinAlgError:
            print(f"Warning: Q_uu singular at step {t}, using pseudo-inverse.")
            Q_uu_inv = np.linalg.pinv(Q_uu_reg)

        K_feedback[t] = -Q_uu_inv @ Q_ux
        k_feedforward[t] = -Q_uu_inv @ Q_u

        P[t] = (
            Q_xx
            + K_feedback[t].T @ Q_uu @ K_feedback[t]
            + K_feedback[t].T @ Q_ux
            + Q_ux.T @ K_feedback[t]
        )
        p[t] = (
            Q_x
            + K_feedback[t].T @ Q_uu @ k_feedforward[t]
            + K_feedback[t].T @ Q_u
            + Q_ux.T @ k_feedforward[t]
        )

    x_path = np.zeros((N + 1, n_states))
    u_path = np.zeros((N, n_controls))
    x_path[0] = start_state

    for t in range(N):
        u_path[t] = K_feedback[t] @ x_path[t] + k_feedforward[t]
        x_path[t + 1] = particle_dynamics(x_path[t], u_path[t])

    return x_path, u_path


def solve_trajectory_generation(x_k, v_k, start_state, goal_pos, obstacles, rho):
    r = cp.Variable((N + 1, 4))
    cost = 0.0
    constraints = [r[0] == start_state]

    terminal_cost = 10000.0 * cp.sum_squares(r[N, :2] - goal_pos)
    cost += terminal_cost

    tracking_term = (rho / 2.0) * cp.sum_squares(x_k - r + v_k)
    cost += tracking_term

    accel_cost = 0.1 * cp.sum_squares(r[1:, 2:] - r[:-1, 2:])
    cost += accel_cost

    for t in range(N):
        constraints.append(r[t + 1, 0] == r[t, 0] + r[t, 2] * DT)
        constraints.append(r[t + 1, 1] == r[t, 1] + r[t, 3] * DT)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    solvers_to_try = [cp.OSQP, cp.SCS]
    solved = False
    for solver in solvers_to_try:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                print(f"  ... Planner (Layer 1) solved with {solver}.")
                solved = True
                break
            else:
                print(
                    f"  ... Planner (Layer 1) tried {solver}, but status was: {prob.status}"
                )
        except cp.error.SolverError:
            print(f"  ... Planner (Layer 1) failed: Solver {solver} is not installed.")
        except Exception as e:
            print(f"  ... Planner (Layer 1) error with {solver}: {e}")

    if not solved:
        print("!!! CRITICAL: All CVXPY solvers failed. Returning previous path.")
        return x_k

    return r.value


def create_obstacle_from_points(points, size=(800, 600)):
    surf = pygame.Surface(size, flags=pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    try:
        pygame.draw.polygon(surf, COLOR_OBSTACLE + (255,), points)
    except Exception:
        return None
    mask = pygame.mask.from_surface(surf)
    return {"poly": points, "mask": mask, "surf": surf}


def create_obstacle_from_shape(
    shape, start=None, end=None, points=None, size=(800, 600)
):
    if shape == "rect" or shape == "oval":
        if start is None or end is None:
            return None
        x1, y1 = start
        x2, y2 = end
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        if right - left < 2 or bottom - top < 2:
            return None
        rect = pygame.Rect(left, top, right - left, bottom - top)
        surf = pygame.Surface(size, flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        if shape == "rect":
            pygame.draw.rect(surf, COLOR_OBSTACLE + (255,), rect)
            poly = [(left, top), (right, top), (right, bottom), (left, bottom)]
        else:
            pygame.draw.ellipse(surf, COLOR_OBSTACLE + (255,), rect)
            poly = [(left, top), (right, top), (right, bottom), (left, bottom)]
        mask = pygame.mask.from_surface(surf)
        return {"poly": poly, "mask": mask, "surf": surf}

    if shape == "triangle":
        if not points or len(points) < 3:
            return None
        surf = pygame.Surface(size, flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        try:
            pygame.draw.polygon(surf, COLOR_OBSTACLE + (255,), points[:3])
        except Exception:
            return None
        mask = pygame.mask.from_surface(surf)
        return {"poly": points[:3], "mask": mask, "surf": surf}
    return None


def point_in_any_obstacle(x, y, obstacles, margin=0):
    xi = int(round(x))
    yi = int(round(y))
    if not (0 <= xi < WIDTH and 0 <= yi < HEIGHT):
        return False
    if margin <= 0:
        for obs in obstacles:
            try:
                if obs["mask"].get_at((xi, yi)):
                    return True
            except (IndexError, KeyError):
                continue
        return False
    m = int(np.ceil(margin))
    x0 = max(0, xi - m)
    x1 = min(WIDTH - 1, xi + m)
    y0 = max(0, yi - m)
    y1 = min(HEIGHT - 1, yi + m)
    for obs in obstacles:
        try:
            for xx in range(x0, x1 + 1):
                for yy in range(y0, y1 + 1):
                    if obs["mask"].get_at((xx, yy)):
                        return True
        except Exception:
            continue
    return False


def point_near_any_obstacle(x, y, obstacles, near_margin=12):
    return point_in_any_obstacle(x, y, obstacles, margin=near_margin)


def obstacle_overlaps_any(candidate_obs, obstacles):
    if candidate_obs is None or "mask" not in candidate_obs:
        return False
    cmask = candidate_obs.get("mask")
    if cmask is None:
        return False
    for ex in obstacles:
        try:
            emask = ex.get("mask")
            if emask is None:
                continue
            if cmask.overlap(emask, (0, 0)) is not None:
                return True
        except Exception:
            continue
    return False


def project_path_onto_constraints(r_path, obstacles, inflation=3):
    r_new = r_path.copy()
    push_applied_ever = False

    ENV_LEFT = 0 + BOUND_MARGIN
    ENV_TOP = 0 + BOUND_MARGIN
    ENV_RIGHT = WIDTH - BOUND_MARGIN
    ENV_BOTTOM = HEIGHT - BOUND_MARGIN

    for _ in range(10):
        push_applied_this_iter = False
        push_vectors = np.zeros_like(r_new[:, :2])

        for t in range(1, N):
            px, py = r_new[t, 0], r_new[t, 1]

            max_push_dist_sq = -1.0
            strongest_push_vec = np.zeros(2)

            for obs in obstacles:
                try:
                    is_inside_or_near = False
                    xi, yi = int(round(px)), int(round(py))
                    if (
                        0 <= xi < WIDTH
                        and 0 <= yi < HEIGHT
                        and obs["mask"].get_at((xi, yi))
                    ):
                        is_inside_or_near = True
                    elif point_in_any_obstacle(px, py, [obs], margin=inflation):
                        is_inside_or_near = True

                    if is_inside_or_near:
                        polys = obs["poly"]
                        cx = sum(p[0] for p in polys) / len(polys)
                        cy = sum(p[1] for p in polys) / len(polys)
                        dx = px - cx
                        dy = py - cy
                        dist_sq = dx * dx + dy * dy
                        norm = np.sqrt(dist_sq)
                        if norm < 1e-3:
                            dx, dy = 1.0, 0.0
                            norm = 1.0

                        max_dx = max(abs(p[0] - cx) for p in polys)
                        max_dy = max(abs(p[1] - cy) for p in polys)
                        radius_approx = np.hypot(max_dx, max_dy)

                        push_needed = max(0, (radius_approx + inflation) - norm) + 1

                        current_push_vec = np.array(
                            [(dx / norm) * push_needed, (dy / norm) * push_needed]
                        )
                        push_dist_sq = (
                            current_push_vec[0] ** 2 + current_push_vec[1] ** 2
                        )

                        if push_dist_sq > max_push_dist_sq:
                            max_push_dist_sq = push_dist_sq
                            strongest_push_vec = current_push_vec

                except Exception:
                    continue

            if max_push_dist_sq > 0:
                push_vectors[t] = strongest_push_vec
                push_applied_this_iter = True
                push_applied_ever = True

        if not push_applied_this_iter and _ > 0:
            break

        r_new[1:N, :2] += push_vectors[1:N, :2]

        r_smoothed_pos = r_new[:, :2].copy()
        for t in range(1, N):
            r_smoothed_pos[t] = (
                0.99 * r_new[t, :2]
                + 0.005 * r_new[t - 1, :2]
                + 0.005 * r_new[t + 1, :2]
            )
        r_new[:, :2] = r_smoothed_pos

    if push_applied_ever:
        r_new[:, 2:] = r_path[:, 2:]

    r_new[:, 0] = np.clip(r_new[:, 0], ENV_LEFT, ENV_RIGHT)
    r_new[:, 1] = np.clip(r_new[:, 1], ENV_TOP, ENV_BOTTOM)

    return r_new


def solve_admm_ocp(start_pos, goal_pos, obstacles):
    if point_in_any_obstacle(
        start_pos[0], start_pos[1], obstacles
    ) or point_in_any_obstacle(goal_pos[0], goal_pos[1], obstacles):
        print("!!! ERROR: Start or Goal is inside an obstacle. No path can be found.")
        return [], [], "Failed: Start/Goal in Obstacle"

    RHO = 10.0
    K_ITER = 30
    CONVERGENCE_THRESHOLD = 15.0

    n_x = 4
    n_u = 2

    r = np.zeros((N + 1, n_x))
    r[:, 0] = np.linspace(start_pos[0], goal_pos[0], N + 1)
    r[:, 1] = np.linspace(start_pos[1], goal_pos[1], N + 1)

    x = r.copy()
    u = np.zeros((N, n_u))
    v = np.zeros((N + 1, n_x))

    start_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])

    history_r = [r[:, :2].copy()]
    history_x = [x[:, :2].copy()]

    converged = False
    status = "Max Iterations Reached"
    for k in range(K_ITER):
        print(f"ADMM Iteration {k+1}/{K_ITER}")
        r_proposed = solve_trajectory_generation(
            x, v, start_state, goal_pos, obstacles, RHO
        )
        if r_proposed is x:
            print("!!! Planner failed, using previous plan. Stopping ADMM.")
            status = "Failed: Planner Error"
            if not history_r:
                history_r.append(r[:, :2].copy())
            if not history_x:
                history_x.append(x[:, :2].copy())
            break

        r = project_path_onto_constraints(r_proposed, obstacles)
        x, u = solve_feedback_control(r, v, start_state, RHO, obstacles=obstacles)
        v = v + (x - r)

        history_r.append(r[:, :2].copy())
        history_x.append(x[:, :2].copy())

        primal_residual = np.linalg.norm(x - r)
        print(f"  ... Primal Residual: {primal_residual:.4f}")

        if np.isnan(primal_residual):
            print("!!! ERROR: Solver diverged to NaN. Stopping.")
            status = "Failed: NaN"
            if history_r:
                history_r.pop()
            if history_x:
                history_x.pop()
            break

        if primal_residual < CONVERGENCE_THRESHOLD and k > 5:
            print("Converged!")
            converged = True
            status = "Converged"
            break

    try:
        final_r = history_r[-1] if history_r else None
        final_x = history_x[-1] if history_x else None
        hit = False
        if final_r is not None:
            for p in final_r:
                if point_in_any_obstacle(p[0], p[1], obstacles):
                    hit = True
                    break
        if not hit and final_x is not None:
            for p in final_x:
                if point_in_any_obstacle(p[0], p[1], obstacles):
                    hit = True
                    break
        if hit:
            print("No solution: final path intersects an obstacle.")
            status = "Failed: Path intersects obstacle"
    except Exception:
        pass

    return history_r, history_x, status


COLOR_BG = (255, 255, 255)
COLOR_START = (0, 0, 255)
COLOR_GOAL = (0, 255, 0)
COLOR_OBSTACLE = (200, 0, 0)
COLOR_PLAN = (255, 165, 0)
COLOR_ROBOT = (0, 150, 0)
COLOR_TEXT = (0, 0, 0)
COLOR_PREVIEW = (0, 150, 0)

BOUND_MARGIN = 6
FIX_START_AT_ORIGIN = True

if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ADMM 2DOF Controller Showcase")
    FONT = pygame.font.SysFont(None, 20)

    ENV_LEFT = 0 + BOUND_MARGIN
    ENV_TOP = 0 + BOUND_MARGIN
    ENV_RIGHT = WIDTH - BOUND_MARGIN
    ENV_BOTTOM = HEIGHT - BOUND_MARGIN

    ORIGIN = (ENV_LEFT + 20, ENV_BOTTOM - 20)

    try:
        rocket_img_orig = pygame.image.load("rocket.png").convert_alpha()
        rocket_img_orig = pygame.transform.scale(rocket_img_orig, (25, 50))
        print("Loaded rocket.png")
    except Exception:
        print("Could not load rocket.png, creating default sprite.")
        rocket_img_orig = pygame.Surface((20, 40), pygame.SRCALPHA)
        rocket_img_orig.fill((0, 0, 0, 0))
        pygame.draw.polygon(
            rocket_img_orig, (200, 200, 255), [(10, 0), (0, 40), (20, 40)]
        )
        pygame.draw.polygon(
            rocket_img_orig, (255, 100, 0), [(5, 35), (0, 40), (10, 40)]
        )
        pygame.draw.polygon(
            rocket_img_orig, (255, 100, 0), [(15, 35), (10, 40), (20, 40)]
        )

    running = True
    start_pos = None
    goal_pos = None
    obstacles = []
    drawing_obstacle = False
    obstacle_start_pos = None
    shape_mode = "oval"

    game_state = "idle"
    solve_status = ""

    history_r = None
    history_x = None
    final_r = None
    final_x = None

    conv_anim_frame = 0
    conv_anim_frame_timer = 0.0
    CONV_FRAME_DURATION = 0.1

    rocket_anim_t = 0.0
    ROCKET_SPEED_FACTOR = 15.0

    if FIX_START_AT_ORIGIN:
        start_pos = ORIGIN

    clock = pygame.time.Clock()

    while running:
        dt_frame = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game state.")
                start_pos, goal_pos = None, None
                obstacles = []
                history_r, history_x, final_r, final_x = None, None, None, None
                drawing_obstacle = False
                game_state = "idle"
                if FIX_START_AT_ORIGIN:
                    start_pos = ORIGIN
                continue

            if game_state == "idle":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if not start_pos:
                            mx, my = event.pos
                            if (
                                ENV_LEFT <= mx <= ENV_RIGHT
                                and ENV_TOP <= my <= ENV_BOTTOM
                            ):
                                start_pos = event.pos
                        elif not goal_pos:
                            mx, my = event.pos
                            if (
                                ENV_LEFT <= mx <= ENV_RIGHT
                                and ENV_TOP <= my <= ENV_BOTTOM
                            ):
                                goal_pos = event.pos
                            else:
                                print("Goal must be inside bounding box.")
                        else:
                            drawing_obstacle = True
                            obstacle_start_pos = event.pos

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and drawing_obstacle:
                        drawing_obstacle = False
                        end_pos = event.pos
                        sx, sy = obstacle_start_pos
                        ex, ey = end_pos
                        sx = max(ENV_LEFT, min(ENV_RIGHT, sx))
                        ex = max(ENV_LEFT, min(ENV_RIGHT, ex))
                        sy = max(ENV_TOP, min(ENV_BOTTOM, sy))
                        ey = max(ENV_TOP, min(ENV_BOTTOM, ey))
                        obs = create_obstacle_from_shape(
                            "oval", start=(sx, sy), end=(ex, ey), size=(WIDTH, HEIGHT)
                        )
                        if obs is not None:
                            if obstacle_overlaps_any(obs, obstacles):
                                print("Skipped obstacle: overlaps existing obstacle.")
                            else:
                                obstacles.append(obs)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if start_pos and goal_pos:
                            print("Requesting solver run...")
                            game_state = "solving"
                            history_r, history_x, final_r, final_x = (
                                None,
                                None,
                                None,
                                None,
                            )

        if game_state == "solving":
            screen.fill(COLOR_BG)
            if start_pos:
                pygame.draw.circle(screen, COLOR_START, start_pos, 10)
            if goal_pos:
                pygame.draw.circle(screen, COLOR_GOAL, goal_pos, 10)
            for obs in obstacles:
                try:
                    screen.blit(obs["surf"], (0, 0))
                except Exception:
                    pass
            text_surf = FONT.render("Status: Solver running...", True, COLOR_TEXT)
            screen.blit(text_surf, (8, 8 + 4 * 18))
            pygame.display.flip()

            start_time = time.time()
            history_r, history_x, solve_status = solve_admm_ocp(
                start_pos, goal_pos, obstacles
            )
            print(
                f"Solver finished in {time.time() - start_time:.2f} seconds. Status: {solve_status}"
            )

            if history_r and history_x and "Failed" not in solve_status:
                final_r = history_r[-1]
                final_x = history_x[-1]
                conv_anim_frame = 0
                conv_anim_frame_timer = 0.0
                game_state = "animating_convergence"
            else:
                print("Solver failed or returned no data, returning to idle.")
                game_state = "idle"

        elif game_state == "animating_convergence":
            conv_anim_frame_timer += dt_frame
            if conv_anim_frame_timer >= CONV_FRAME_DURATION:
                conv_anim_frame_timer -= CONV_FRAME_DURATION
                conv_anim_frame += 1
                if not history_r or conv_anim_frame >= len(history_r):
                    rocket_anim_t = 0.0
                    game_state = "animating_rocket"
                    conv_anim_frame = len(history_r) - 1 if history_r else 0

        elif game_state == "animating_rocket":
            rocket_anim_t += ROCKET_SPEED_FACTOR * dt_frame
            if rocket_anim_t >= N:
                game_state = "solved_static"
                rocket_anim_t = N

        screen.fill(COLOR_BG)

        pygame.draw.rect(
            screen,
            (50, 50, 50),
            (ENV_LEFT, ENV_TOP, ENV_RIGHT - ENV_LEFT, ENV_BOTTOM - ENV_TOP),
            2,
        )

        if start_pos:
            pygame.draw.circle(screen, COLOR_START, start_pos, 10)
        if start_pos:
            orig_label = FONT.render("ORIGIN", True, COLOR_TEXT)
            screen.blit(orig_label, (start_pos[0] + 8, start_pos[1] - 8))
        if goal_pos:
            pygame.draw.circle(screen, COLOR_GOAL, goal_pos, 10)

        for obs in obstacles:
            try:
                screen.blit(obs["surf"], (0, 0))
            except Exception:
                pass

        if game_state == "idle":
            mouse_pos = pygame.mouse.get_pos()
            if drawing_obstacle and obstacle_start_pos:
                sx, sy = obstacle_start_pos
                ex, ey = mouse_pos
                x = min(sx, ex)
                y = min(sy, ey)
                w = abs(ex - sx)
                h = abs(ey - sy)
                preview_rect = pygame.Rect(x, y, w, h)
                pygame.draw.ellipse(screen, COLOR_PREVIEW, preview_rect, 2)

        if game_state == "animating_convergence":
            frame_idx = 0
            num_frames = 0
            if history_r:
                num_frames = len(history_r)
                frame_idx = min(conv_anim_frame, num_frames - 1)
            if history_r:
                current_r = history_r[frame_idx]
                try:
                    if len(current_r) > 1:
                        pts_r = [tuple(map(int, p)) for p in current_r]
                        pygame.draw.lines(screen, COLOR_PLAN, False, pts_r, 3)
                except Exception as e:
                    print(f"Error drawing r path: {e}")
            if history_x:
                frame_idx_x = min(frame_idx, len(history_x) - 1)
                current_x = history_x[frame_idx_x]
                try:
                    if len(current_x) > 1:
                        pts_x = [tuple(map(int, p)) for p in current_x]
                        pygame.draw.lines(screen, COLOR_ROBOT, False, pts_x, 4)
                except Exception as e:
                    print(f"Error drawing x path: {e}")
            iter_text = f"ADMM Iteration: {frame_idx + 1}/{num_frames}"
            surf = FONT.render(iter_text, True, COLOR_TEXT)
            screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, 8))

        elif game_state == "animating_rocket":
            if final_r is not None:
                try:
                    if len(final_r) > 1:
                        pts_r = [tuple(map(int, p)) for p in final_r]
                        pygame.draw.lines(screen, COLOR_PLAN, False, pts_r, 3)
                except Exception as e:
                    print(f"Error drawing final r path: {e}")

            if final_x is not None:
                try:
                    current_idx = min(int(rocket_anim_t), N)
                    if current_idx > 0 and len(final_x) > current_idx:
                        pts_x_trail = [
                            tuple(map(int, p)) for p in final_x[: current_idx + 1]
                        ]
                        pygame.draw.lines(screen, COLOR_ROBOT, False, pts_x_trail, 4)
                except Exception as e:
                    print(f"Error drawing rocket trail: {e}")

            if final_x is not None and rocket_img_orig is not None and len(final_x) > 1:
                try:
                    idx0 = min(int(rocket_anim_t), N)
                    idx1 = min(idx0 + 1, N)
                    if idx0 >= N or idx1 >= len(final_x):
                        idx0 = N - 1
                        idx1 = N
                        if idx1 >= len(final_x):
                            if len(final_x) > 0:
                                current_pos = final_x[-1]
                            else:
                                current_pos = start_pos
                            pos0 = final_x[-2] if len(final_x) > 1 else current_pos
                            pos1 = final_x[-1] if len(final_x) > 0 else current_pos
                        else:
                            current_pos = final_x[N]
                            pos0 = final_x[N - 1]
                            pos1 = final_x[N]
                        frac = 1.0
                    else:
                        frac = rocket_anim_t - idx0
                        pos0 = final_x[idx0]
                        pos1 = final_x[idx1]
                        current_pos = pos0 * (1 - frac) + pos1 * frac

                    dx = pos1[0] - pos0[0]
                    dy = pos1[1] - pos0[1]
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                        if idx0 > 0 and idx0 < len(final_x):
                            prev_pos0 = final_x[idx0 - 1]
                            prev_pos1 = final_x[idx0]
                            dx = prev_pos1[0] - prev_pos0[0]
                            dy = prev_pos1[1] - prev_pos0[1]
                        else:
                            dx, dy = 0, -1

                    angle_rad = np.arctan2(-dx, dy)
                    angle_deg = np.degrees(angle_rad)
                    rotated_rocket = pygame.transform.rotate(rocket_img_orig, angle_deg)
                    rocket_rect = rotated_rocket.get_rect(
                        center=tuple(map(int, current_pos))
                    )
                    screen.blit(rotated_rocket, rocket_rect.topleft)
                except IndexError:
                    print(
                        f"Index Error drawing rocket (rocket_anim_t={rocket_anim_t}, N={N}, len(final_x)={len(final_x) if final_x is not None else 0})"
                    )
                    if final_x is not None and len(final_x) > 0:
                        pygame.draw.circle(
                            screen, (255, 0, 0), tuple(map(int, final_x[-1])), 10
                        )
                except Exception as e:
                    print(f"Error drawing rocket: {e}")

        elif game_state == "solved_static":
            if final_r is not None:
                try:
                    if len(final_r) > 1:
                        pts_r = [tuple(map(int, p)) for p in final_r]
                        pygame.draw.lines(screen, COLOR_PLAN, False, pts_r, 3)
                except Exception as e:
                    print(f"Error drawing final static r: {e}")
            if final_x is not None:
                try:
                    if len(final_x) > 1:
                        pts_x = [tuple(map(int, p)) for p in final_x]
                        pygame.draw.lines(screen, COLOR_ROBOT, False, pts_x, 4)
                except Exception as e:
                    print(f"Error drawing final static x: {e}")

        instr_lines = [
            "Click GOAL",
            "Shape mode: OVAL",
            "After start+goal: draw obstacles (drag)",
            "SPACE: run solver   R: reset",
        ]
        if game_state == "solving":
            status_text = "Status: Solver running..."
        elif game_state == "animating_convergence":
            status_text = f"Status: Animating Convergence (Iter {conv_anim_frame+1})"
        elif game_state == "animating_rocket":
            status_text = "Status: Animating Rocket..."
        elif game_state == "solved_static":
            status_text = f"Status: Solved! ({solve_status})"
        elif start_pos and goal_pos:
            status_text = "Status: Ready to solve (SPACE)"
        else:
            status_text = "Status: Set start and goal"
        instr_lines.append(status_text)

        for i, line in enumerate(instr_lines):
            surf = FONT.render(line, True, COLOR_TEXT)
            screen.blit(surf, (8, 8 + i * 18))

        pygame.display.flip()

    pygame.quit()
