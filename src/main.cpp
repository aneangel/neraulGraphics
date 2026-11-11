#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <SDL2/SDL.h>
#include <neraulgraphics/renderer.h>

using namespace neraulgraphics;

namespace {
    // The library's public Vec3 is plain data, so the viewer keeps the little
    // vector math its camera controls need.
    Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    Vec3 operator*(const Vec3 &v, const float s) { return {v.x * s, v.y * s, v.z * s}; }
    Vec3 &operator+=(Vec3 &a, const Vec3 &b) { return a = a + b; }
    Vec3 &operator-=(Vec3 &a, const Vec3 &b) { return a = a - b; }

    Vec3 cross(const Vec3 &a, const Vec3 &b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    Vec3 normalize(const Vec3 &v) { return v * (1.0f / std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z)); }

    Vec3 forward_from_angles(const float yaw, const float pitch) {
        return {cosf(pitch) * sinf(yaw), sinf(pitch), -cosf(pitch) * cosf(yaw)};
    }

    // Rotation of `angle` radians about a unit axis, as the quaternion make_transform takes.
    Quat axis_angle(const Vec3 &axis, const float angle) {
        const float s = std::sin(0.5f * angle);
        return {axis.x * s, axis.y * s, axis.z * s, std::cos(0.5f * angle)};
    }

    // Deterministic 0..1 values, so the same --instances N always makes the same scene.
    float hash01(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
        return static_cast<float>(x >> 8) / static_cast<float>(1 << 24);
    }

    // One spinning, bobbing body. The mesh is shared; only this record is per-body.
    struct Body {
        Vec3 position;
        Vec3 axis;
        float spin;   // radians per second
        float scale;
        float phase;
    };

    std::vector<Body> make_bodies(const int count) {
        std::vector<Body> bodies;
        bodies.reserve(count);
        for (int i = 0; i < count; ++i) {
            const uint32_t k = static_cast<uint32_t>(i) * 8u;
            Vec3 axis = {hash01(k + 3) - 0.5f, hash01(k + 4) - 0.5f, hash01(k + 5) - 0.5f};
            axis = normalize(axis + Vec3{0.0f, 0.05f, 0.0f});
            bodies.push_back({{-25.0f + 50.0f * hash01(k), -3.0f + 15.0f * hash01(k + 1), -12.0f - 60.0f * hash01(k + 2)},
                axis, 0.5f + 2.0f * hash01(k + 6), 0.3f + 0.5f * hash01(k + 7), 6.28f * hash01(k + 8)});
        }
        return bodies;
    }

    Transform body_transform(const Body &b, const float time) {
        const Vec3 p = {b.position.x, b.position.y + 0.5f * std::sin(time + b.phase), b.position.z};
        return make_transform(p, axis_angle(b.axis, b.spin * time + b.phase), b.scale);
    }

    // What the window shows: the picture, or one of the sensor images, coloured so a person can read it.
    enum class View { Rgb, Depth, Normals, Ids };

    const char *view_name(const View v) {
        switch (v) {
            case View::Depth: return "depth";
            case View::Normals: return "normals";
            case View::Ids: return "object ids";
            default: return "rgb";
        }
    }

    uint32_t pack_argb(const float r, const float g, const float b) {
        auto byte = [](const float c) { return static_cast<uint32_t>(std::max(0.0f, std::min(1.0f, c)) * 255.0f + 0.5f); };
        return 0xFF000000u | byte(r) << 16 | byte(g) << 8 | byte(b);
    }

    // This file is built with -ffast-math, under which the compiler may assume there are no infinities,
    // so "is this depth infinite" is answered by reading the float's exponent bits.
    bool is_finite_value(const float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x7f800000u) != 0x7f800000u;
    }

    // A sensor image as 0xAARRGGBB pixels. Depth is white up close and fades to black far away, with
    // nothing hit black. Normals are the usual colour code (x, y, z as red, green, blue), so a surface
    // facing the camera is blue and one facing up is green. Each object ID gets its own colour.
    void colorize(const HostFrame &f, const View view, std::vector<uint32_t> &out) {
        const size_t n = static_cast<size_t>(f.width) * f.height;
        out.assign(n, 0xFF000000u);
        if (view == View::Depth) {
            std::vector<float> finite;
            for (const float d : f.depth) if (is_finite_value(d)) finite.push_back(d);
            if (finite.empty()) return;
            const float near_d = *std::min_element(finite.begin(), finite.end());
            // The 98th percentile, not the maximum, so one far-away speck does not wash out everything else.
            const size_t k = finite.size() * 98 / 100;
            std::nth_element(finite.begin(), finite.begin() + static_cast<std::ptrdiff_t>(k), finite.end());
            const float far_d = std::max(finite[k], near_d + 1e-3f);
            for (size_t i = 0; i < n; ++i)
                if (is_finite_value(f.depth[i])) {
                    const float c = 1.0f - (f.depth[i] - near_d) / (far_d - near_d);
                    out[i] = pack_argb(c, c, c);
                }
        } else if (view == View::Normals) {
            for (size_t i = 0; i < n; ++i)
                out[i] = pack_argb(f.normal[3 * i] * 0.5f + 0.5f, f.normal[3 * i + 1] * 0.5f + 0.5f, f.normal[3 * i + 2] * 0.5f + 0.5f);
        } else if (view == View::Ids) {
            for (size_t i = 0; i < n; ++i) {
                const uint32_t id = static_cast<uint32_t>(f.object_id[i]);
                if (id == static_cast<uint32_t>(OBJECT_NONE)) continue;
                const uint32_t h = id * 2654435761u;  // spreads neighbouring IDs across the colour wheel
                out[i] = pack_argb(0.25f + 0.75f * static_cast<float>(h >> 24 & 0xFF) / 255.0f,
                    0.25f + 0.75f * static_cast<float>(h >> 16 & 0xFF) / 255.0f, 0.25f + 0.75f * static_cast<float>(h >> 8 & 0xFF) / 255.0f);
            }
        }
    }

    unsigned outputs_for(const View v) {
        return v == View::Depth ? OUTPUT_DEPTH : v == View::Normals ? OUTPUT_NORMAL : v == View::Ids ? OUTPUT_OBJECT_ID : OUTPUT_RGB;
    }

    void save_ppm(const uint32_t *pixels, const int width, const int height, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        ofs << "P6\n" << width << " " << height << "\n255\n";
        std::vector<char> rgb(static_cast<size_t>(width) * height * 3);
        for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
            rgb[i * 3 + 0] = static_cast<char>(pixels[i] >> 16 & 0xFFu);
            rgb[i * 3 + 1] = static_cast<char>(pixels[i] >> 8 & 0xFFu);
            rgb[i * 3 + 2] = static_cast<char>(pixels[i] & 0xFFu);
        }
        ofs.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    }
}

int run(const int instance_count, View view, const std::string &snapshot, bool beauty) {
    constexpr int width = 1920;
    constexpr int height = 1080;
    constexpr float fov = M_PI / 2;
    // Rays per pixel each frame. Raise to anti-alias while the camera is moving.
    constexpr int samples_per_frame = 1;
    const Vec3 world_up = Vec3{0, 1, 0};

    // colour, diffuse albedo, specular albedo, reflectivity, specular exponent
    const Material ivory = make_material(Vec3{0.4f, 0.4f, 0.3f}, 0.6f, 0.3f, 0.1f, 50.0f);
    const Material red_rubber = make_material(Vec3{0.3f, 0.1f, 0.1f}, 0.9f, 0.1f, 0.0f, 10.0f);
    const Material mirror = make_material(Vec3{1.0f, 1.0f, 1.0f}, 0.0f, 10.0f, 0.8f, 1425.0f);
    const Material purple = make_material(Vec3{2.0, 1.0, 4.0}, 0.6, 0.3, 0.1, 50.0f);

    std::vector<Sphere> spheres = {
        {Vec3{-3.0f, 0.0f, -16.0f}, 2.0f, ivory},
        {Vec3{-1.0f, -1.5f, -12.0f}, 2.0f, mirror},
        {Vec3{1.5f, -0.5f, -18.0f}, 3.0f, red_rubber},
        {Vec3{7.0f, 5.0f, -18.0f}, 4.0f, mirror},
        {Vec3{0.4, 0.5, -25.0f}, 3.0f, purple},
    };

    // Last field is the light's radius: 0 gives hard shadows, above 0 gives soft ones.
    std::vector<Light> lights = {
        {Vec3{-20.0f, 20.0f, 20.0f}, 1.5f, 0.0f},
        {Vec3{30.0f, 50.0f, -25.0f}, 1.8f, 0.0f},
        {Vec3{30.0f, 50.0f, 30.0f}, 1.7f, 0.0f},
    };

    Plane floor{};
    floor.y = -4.0f;
    floor.half_extent_x = 10.0f;
    floor.z_min = -30.0f;
    floor.z_max = -10.0f;
    floor.checker_scale = 0.5f;
    floor.material = make_material(Vec3{1.0f, 1.0f, 1.0f}, 1.0f, 0.0f, 0.0f, 1.0f);

    // Spinning shapes for --instances N: two meshes shared by every body, so adding
    // bodies adds instances, not geometry.
    const std::vector<Mesh> meshes = {make_box({1.0f, 1.0f, 1.0f}), make_icosphere(1.0f, 2)};
    const std::vector<Body> bodies = make_bodies(instance_count);
    std::vector<Instance> instances;
    std::vector<Transform> transforms(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        const Vec3 color = {0.2f + 0.8f * hash01(i * 8u + 9u), 0.2f + 0.8f * hash01(i * 8u + 10u), 0.2f + 0.8f * hash01(i * 8u + 11u)};
        instances.push_back({static_cast<int>(i % 2), body_transform(bodies[i], 0.0f),
            make_material(color, 0.8f, 0.3f, 0.0f, 40.0f)});
    }

    // Built before the window, so a bad scene is reported without one flashing up.
    // Frees its device memory when it goes out of scope.
    Renderer gpu_renderer(width, height, spheres, lights, floor, meshes, instances);
    // Beauty mode is path tracing: bounced light, soft shadows from the sky, physically based materials, denoised so a
    // few samples per pixel look clean. The classic picture is the fast one, for flying about.
    auto apply_mode = [&] {
        if (beauty) gpu_renderer.set_render_mode(RenderMode::PathTraced, PathTracing{});
        else gpu_renderer.set_render_mode(RenderMode::Classic);
    };
    apply_mode();
    float update_ms = 0.0f;
    float time = 0.0f;

    // --snapshot: draw one frame from the starting camera, save it, and leave, without opening a window.
    if (!snapshot.empty()) {
        const Camera camera = make_camera(Vec3{0, 0, 0}, forward_from_angles(0.0f, 0.0f), world_up, fov, width, height);
        std::vector<uint32_t> shown;
        if (view == View::Rgb) {
            // A path-traced picture refines itself over frames; the classic one just takes 4 rays a pixel.
            const uint32_t *p = nullptr;
            if (beauty) for (int f = 0; f < 64; ++f) p = gpu_renderer.render(camera, f, 1);
            else p = gpu_renderer.render(camera, 0, 4);
            shown.assign(p, p + static_cast<size_t>(width) * height);
        } else {
            colorize(gpu_renderer.render_frame_host(camera, 0, 1, outputs_for(view)), view, shown);
        }
        save_ppm(shown.data(), width, height, snapshot);
        std::cout << "wrote " << snapshot << " (" << view_name(view) << (beauty && view == View::Rgb ? ", path traced" : "") << ", " << width << "x" << height << ")\n";
        return 0;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height, SDL_WINDOW_RESIZABLE);
    // No vsync flag: it would cap the loop at the monitor refresh and hide the real speed.
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, width, height);
    SDL_RenderSetLogicalSize(renderer, width, height);

    Vec3 position = Vec3{0, 0, 0};
    float yaw = 0, pitch = 0;
    bool looking = false, running = true;
    int frame = 0;
    const uint32_t *pixels = nullptr;
    uint64_t prev = SDL_GetPerformanceCounter();

    std::cout << "drag left mouse to look, WASD to move, Q/E down/up, shift to sprint, P to save, Esc to quit\n"
              << "1 rgb   2 depth   3 normals   4 object ids   (what a depth camera, a surface-normal sensor and a segmenter see)\n"
              << "B beauty mode: path-traced light, soft sky shadows and bounces, denoised (hold still and it keeps improving)\n";
    bool mode_changed = false;
    std::vector<uint32_t> shown;  // a sensor image as it is displayed

    while (running) {
        const uint64_t now = SDL_GetPerformanceCounter();
        const float dt = std::max(1e-4f, std::min(0.1f,
            static_cast<float>(now - prev) / static_cast<float>(SDL_GetPerformanceFrequency())));
        prev = now;

        const Vec3 prev_position = position;
        const float prev_yaw = yaw, prev_pitch = pitch;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) looking = true;
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT) looking = false;
                    break;
                case SDL_MOUSEMOTION:
                    if (looking) {
                        yaw += static_cast<float>(event.motion.xrel) * 0.004f;
                        // Must stay short of +/-pi/2, where the camera basis degenerates.
                        pitch = std::max(-1.55f, std::min(1.55f,
                            pitch - static_cast<float>(event.motion.yrel) * 0.004f));
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
                    if (event.key.keysym.scancode == SDL_SCANCODE_1) view = View::Rgb;
                    if (event.key.keysym.scancode == SDL_SCANCODE_2) view = View::Depth;
                    if (event.key.keysym.scancode == SDL_SCANCODE_3) view = View::Normals;
                    if (event.key.keysym.scancode == SDL_SCANCODE_4) view = View::Ids;
                    if (event.key.keysym.scancode == SDL_SCANCODE_B) {
                        beauty = !beauty;
                        apply_mode();
                        mode_changed = true;
                    }
                    if (event.key.keysym.scancode == SDL_SCANCODE_P && pixels) {
                        save_ppm(pixels, width, height, "./screenshot.ppm");
                        std::cout << "wrote ./screenshot.ppm (" << frame + 1 << " samples/pixel)\n";
                    }
                    break;
                default:
                    break;
            }
        }

        const Uint8 *keys = SDL_GetKeyboardState(nullptr);
        const float speed = (keys[SDL_SCANCODE_LSHIFT] ? 24.f : 8.f) * dt;
        const Vec3 forward = forward_from_angles(yaw, pitch);
        const Vec3 right = normalize(cross(forward, world_up));
        if (keys[SDL_SCANCODE_W]) position += forward * speed;
        if (keys[SDL_SCANCODE_S]) position -= forward * speed;
        if (keys[SDL_SCANCODE_A]) position -= right * speed;
        if (keys[SDL_SCANCODE_D]) position += right * speed;
        if (keys[SDL_SCANCODE_E]) position += world_up * speed;
        if (keys[SDL_SCANCODE_Q]) position -= world_up * speed;

        // Any camera change invalidates the accumulated samples, so the average restarts.
        bool moved = position.x != prev_position.x || position.y != prev_position.y ||
            position.z != prev_position.z || yaw != prev_yaw || pitch != prev_pitch;

        // Moving bodies changes the picture too, so with any instances the image never settles.
        if (!bodies.empty()) {
            time += dt;
            const auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < bodies.size(); ++i) transforms[i] = body_transform(bodies[i], time);
            gpu_renderer.set_instance_transforms(transforms);
            update_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
            moved = true;
        }
        // Classic pictures average while the camera holds still, so any change restarts them. A denoised path-traced
        // picture instead carries its history along with the camera, so its frame number only restarts with a new mode.
        frame = mode_changed ? 0 : (moved && !beauty) ? 0 : frame + 1;
        mode_changed = false;

        const Camera camera = make_camera(position, position + forward_from_angles(yaw, pitch),
            world_up, fov, width, height);

        if (view == View::Rgb) {
            pixels = gpu_renderer.render(camera, frame, samples_per_frame);
        } else {
            // The sensor images are one exact ray per pixel, so there is nothing to accumulate.
            colorize(gpu_renderer.render_frame_host(camera, 0, 1, outputs_for(view)), view, shown);
            pixels = shown.data();
        }

        SDL_UpdateTexture(texture, nullptr, pixels, width * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        char title[200];
        snprintf(title, sizeof(title), "raytracer [%s%s]  %dx%d  %.1f fps  kernel %.2f ms  %d spp  pos %.1f %.1f %.1f",
            view_name(view), beauty && view == View::Rgb ? ", beauty" : "", width, height, 1.f / dt, gpu_renderer.last_kernel_ms(),
            view == View::Rgb ? (frame + 1) * samples_per_frame : 1, position.x, position.y, position.z);
        if (!bodies.empty()) {
            const size_t used = strlen(title);
            snprintf(title + used, sizeof(title) - used, "  %zu bodies  update %.2f ms", bodies.size(), update_ms);
        }
        SDL_SetWindowTitle(window, title);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

// The renderer reports failures as exceptions; here they become a message and an exit status.
int main(const int argc, char **argv) {
    int instance_count = 0;
    View view = View::Rgb;
    std::string snapshot;
    bool beauty = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--instances" && i + 1 < argc) {
            instance_count = std::atoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "rgb") view = View::Rgb;
            else if (m == "depth") view = View::Depth;
            else if (m == "normals") view = View::Normals;
            else if (m == "ids") view = View::Ids;
            else {
                std::cerr << "error: --mode must be rgb, depth, normals or ids\n";
                return 1;
            }
        } else if (arg == "--snapshot" && i + 1 < argc) {
            snapshot = argv[++i];
        } else if (arg == "--beauty") {
            beauty = true;
        } else {
            std::cerr << "usage: " << argv[0] << " [--instances N] [--mode rgb|depth|normals|ids] [--snapshot FILE.ppm] [--beauty]\n"
                      << "  --instances N        add N spinning boxes and spheres, to see how the BVH copes\n"
                      << "  --mode M             start showing the picture (rgb) or a sensor image: depth, normals or ids\n"
                      << "                       (keys 1 to 4 switch while running)\n"
                      << "  --snapshot FILE.ppm  save one frame and exit, without opening a window\n"
                      << "  --beauty             start in beauty mode: path-traced light, soft sky shadows and bounces, denoised\n"
                      << "                       (key B switches while running)\n";
            return arg == "--help" ? 0 : 1;
        }
    }
    if (instance_count < 0) {
        std::cerr << "error: --instances must be 0 or more\n";
        return 1;
    }

    try {
        return run(instance_count, view, snapshot, beauty);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
