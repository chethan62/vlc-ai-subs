--[[
vlc-ai-subs — VLC extension for AI-powered subtitle generation.

Compatible with VLC 3.x and VLC 4.x.

Engines: Auto (Parakeet for English, WhisperX otherwise) / WhisperX / Parakeet.

Two modes (Generate & Load is the dialog default):
  1. Generate & Load — full SRT is created next to the media, then loaded synced to playback
  2. Real-time OSD  — SRT goes to a temp path; each cue is pushed to the OSD as it is produced

The dialog remembers your last choices (engine/model/language/task/mode) in
<vlc user data dir>/vlc-ai-subs/settings.conf, and a run can be cancelled with
the Cancel button (the CLI records its PID in <mirror>.pid; we signal it and it
stops the ML child — VLC's Lua has no process API).

Requires: Python 3.12 + WhisperX (word-level aligned subtitles)
Install:  Run setup.sh (Linux/macOS) or setup.bat (Windows).

https://github.com/chethan62/vlc-ai-subs
]]

function descriptor()
    return {
        title = "AI Subs Generator",
        version = "3.4",
        author = "chethan62",
        url = "https://github.com/chethan62/vlc-ai-subs",
        shortdesc = "AI subtitle generator (WhisperX/Parakeet)",
        description = "Generate subtitles using local AI. "
            .. "WhisperX (multilingual) or Parakeet (English, fastest). "
            .. "Real-time OSD or generate-and-load SRT. "
            .. "Compatible with VLC 3.x and 4.x.",
        capabilities = {"menu"},
    }
end

local dlg            = nil
local engine_dropdown = nil
local model_dropdown  = nil
local lang_input      = nil
local task_dropdown   = nil
local mode_dropdown   = nil
local status_label    = nil
local details_label   = nil
local cue_label       = nil
local progress_bar    = nil
local debug_label     = nil
local osd_channel     = nil

-- Dropdown value tables. The FIRST item is the default (VLC combo boxes select
-- the first added item and there is no dropdown:set_value()), so restoring a
-- saved choice = adding that item first, keeping the mapping id-keyed.
local ENGINES = {
    { "auto",       "Auto (Parakeet for English, else best GPU engine)" },
    { "whisperx",   "WhisperX (multilingual, aligned)" },
    { "parakeet",   "Parakeet (English, fastest)" },
    { "whispercpp", "whisper.cpp (Vulkan - AMD/Intel GPUs)" },
}
local MODELS = {
    { "recommended",    "Recommended (auto)" },
    { "tiny",           "tiny (fastest)" },
    { "base",           "base (balanced)" },
    { "small",          "small (accurate)" },
    { "medium",         "medium (very accurate)" },
    { "large",          "large (best quality)" },
    { "large-v3-turbo", "large-v3-turbo (fast + accurate)" },
}
local TASKS = {
    { "translate",  "Translate to English" },
    { "transcribe", "Transcribe (same language)" },
}
local MODES = {
    { "srt",      "Generate & Load SRT" },
    { "realtime", "Real-time OSD" },
}
local SETTING_KEYS = { "engine", "model", "language", "task", "mode" }
local engine_map, model_map, task_map, mode_map = {}, {}, {}, {}

-- Polling state (set by start_generation, used by poll_progress)
local _poll_tmp      = nil
local _poll_mode     = nil
local _poll_model    = nil
local _poll_engine   = nil
local _poll_tmr      = nil
local _poll_secs     = 0
local _poll_duration = 0
local _poll_est_total = 30
-- 'sub' events already pushed to the OSD (real-time mode) — cues are shown
-- once, as they are produced, never replayed at the end.
local _poll_shown    = 0
-- Latest status/transcript lines seen in the mirror file (dialog details pane)
local _poll_status   = nil
local _poll_cue      = nil
local _poll_cues     = 0
local POLL_US     = 1000000  -- poll every 1 second (was 3s)

-- Seed the temp-name RNG once at load — predictable /tmp names are a
-- symlink-attack vector (see get_temp_file).
-- Guarded: VLC's extension *scan* runs scripts in a bare lua state with no
-- standard libs (math is nil) — an unguarded call here aborts registration.
-- At runtime GetLuaState() opens all libs, so the seed then actually runs.
if math then
    math.randomseed(os.time() * 1000 + (os.clock() * 1000) % 1000)
end

----------------------------------------------------------------
-- Lifecycle
----------------------------------------------------------------

function activate()
    vlc.msg.info("[AI Subs] activate() called")
    create_dialog()
end
function deactivate() if dlg then dlg:delete(); dlg = nil end end
function close()      deactivate() end

function menu() return {"Generate Subtitles"} end
function trigger_menu(id) if id == 1 then create_dialog() end end

----------------------------------------------------------------
-- Dialog
----------------------------------------------------------------

function create_dialog()
    vlc.msg.info("[AI Subs] create_dialog()")
    -- OSD fallback: always visible even if dialog fails on Wayland
    vlc.osd.message("AI Subs Generator ready — check View menu", 3)

    if dlg then dlg:delete() end
    dlg = vlc.dialog("AI Subs Generator")

    local saved = load_settings()

    dlg:add_label("Engine:", 1, 1, 1, 1)
    engine_dropdown = dlg:add_dropdown(2, 1, 2, 1)
    engine_map = fill_dropdown(engine_dropdown, ENGINES, saved.engine)

    dlg:add_label("Model:", 1, 2, 1, 1)
    model_dropdown = dlg:add_dropdown(2, 2, 2, 1)
    model_map = fill_dropdown(model_dropdown, MODELS, saved.model)

    dlg:add_label("Language:", 1, 3, 1, 1)
    lang_input = dlg:add_text_input(saved.language or "auto", 2, 3, 2, 1)

    dlg:add_label("Task:", 1, 4, 1, 1)
    task_dropdown = dlg:add_dropdown(2, 4, 2, 1)
    task_map = fill_dropdown(task_dropdown, TASKS, saved.task)

    dlg:add_label("Mode:", 1, 5, 1, 1)
    mode_dropdown = dlg:add_dropdown(2, 5, 2, 1)
    mode_map = fill_dropdown(mode_dropdown, MODES, saved.mode)

    -- Generate + Cancel share a row; VLC shrinks dialogs, so every widget gets
    -- an explicit col/row/hspan/vspan.
    dlg:add_button("Generate", start_generation, 1, 6, 2, 1)
    dlg:add_button("Cancel",   cancel_run,       3, 6, 1, 1)

    status_label  = dlg:add_label("Ready. Play a media file and click Generate.", 1, 7, 3, 1)
    progress_bar  = dlg:add_progress_bar(0, 1, 8, 3, 1)
    details_label = dlg:add_label("", 1, 9, 3, 1)
    cue_label     = dlg:add_label("", 1, 10, 3, 1)
    debug_label   = dlg:add_label("", 1, 11, 3, 1)
    dlg:show()
end

----------------------------------------------------------------
-- Dropdown helpers
----------------------------------------------------------------

-- Add `items` to a dropdown with `prefer` FIRST (= the default: VLC selects the
-- item added first, and the dropdown widget has no set_value()). Ids are the
-- dropdown positions, so get_value() maps straight back to the canonical name.
function fill_dropdown(dropdown, items, prefer)
    local order, matched = {}, false
    for i, it in ipairs(items) do
        if prefer and it[1] == prefer then order[#order + 1] = i; matched = true end
    end
    if not matched then order[#order + 1] = 1 end
    for i = 1, #items do
        local dup = false
        for _, o in ipairs(order) do if o == i then dup = true end end
        if not dup then order[#order + 1] = i end
    end
    local map = {}
    for idx, item_index in ipairs(order) do
        dropdown:add_value(items[item_index][2], idx)
        map[idx] = items[item_index][1]
    end
    return map
end

-- Current choice, falling back to the first item (also the -1/"nothing
-- selected" case, which is the restored-or-default item).
local function picked(dropdown, map)
    local id = dropdown:get_value()
    if id and map[id] then return map[id] end
    return map[1]
end

function get_model_name() return picked(model_dropdown, model_map) end
function get_task()       return picked(task_dropdown, task_map) end
function get_mode()       return picked(mode_dropdown, mode_map) end
function get_engine()     return picked(engine_dropdown, engine_map) end

-- Resolve the engine choice to one that can actually do the job: Parakeet has
-- no translation and no language detection, so "Auto" only picks it for an
-- explicit English transcribe run. Everything else stays "auto" and is decided
-- in Python, which is the side that can see the hardware (NVIDIA CUDA →
-- WhisperX; Vulkan-only GPU → whisper.cpp). Returns engine, note.
function engine_for(engine, language, task)
    local english = language and string.match(language, "^en")
    if engine == "parakeet" then
        if task == "translate" then
            return "whisperx", "Parakeet cannot translate - using WhisperX"
        end
        if language and language ~= "auto" and not english then
            return "whisperx", "Parakeet is English-only - using WhisperX"
        end
        return "parakeet", nil
    end
    if engine == "auto" then
        if task == "transcribe" and english then return "parakeet", nil end
        return "auto", nil
    end
    if engine == "whispercpp" and task == "translate" then
        return "whispercpp",
            "whisper.cpp translates with Whisper itself (NLLB cascade needs WhisperX)"
    end
    return engine, nil
end

-- Human label for status lines.
function engine_label(engine)
    if engine == "parakeet" then return "Parakeet" end
    if engine == "whispercpp" then return "whisper.cpp" end
    if engine == "auto" then return "Auto" end
    return "WhisperX"
end

----------------------------------------------------------------
-- VLC version compatibility (3.x / 4.x)
----------------------------------------------------------------

function get_input_item()
    local ok, item
    ok, item = pcall(function() return vlc.player.item() end)
    if ok and item then return item end
    ok, item = pcall(function() return vlc.input.item() end)
    if ok and item then return item end
    return nil
end

function add_subtitle_track(srt_path)
    local ok
    ok = pcall(function() vlc.player.add_subtitle(srt_path) end)
    if ok then return true end
    ok = pcall(function() vlc.input.add_subtitle(srt_path) end)
    if ok then return true end
    ok = pcall(function()
        local input = vlc.object.input()
        if input then vlc.var.set(input, "sub-file", srt_path) end
    end)
    return ok
end

function register_osd()
    local ok, ch = pcall(function() return vlc.osd.channel_register() end)
    if ok and ch then return ch end
    return 1
end

function show_osd(text, duration)
    if not text then return end
    local ok = pcall(function()
        vlc.osd.message(text, osd_channel, "bottom", duration)
    end)
    if not ok then
        pcall(function() vlc.osd.message(text, osd_channel) end)
    end
end

-- Push one transcribed cue to the OSD as soon as it is produced.
-- Duration follows the cue length (min 1.5 s; 3 s when there is no usable
-- timing) — VLC replaces the message on the channel, so the newest cue is
-- what stays on screen while generation continues. Cues carry SRT line
-- breaks; the OSD wants a single line.
function show_cue_osd(d)
    local dur = 3000000
    if d.start and d["end"] then
        dur = math.max((d["end"] - d.start) * 1000000, 1500000)
    end
    osd_channel = osd_channel or register_osd()
    show_osd(string.gsub(d.text or "", "%s+", " "), dur)
end

----------------------------------------------------------------
-- Path helpers
----------------------------------------------------------------

function is_windows()
    return package.config:sub(1, 1) == "\\"
end

function get_home()
    -- USERPROFILE is the standard Windows home directory variable
    local home = os.getenv("USERPROFILE") or os.getenv("HOME") or ""
    return home
end

function get_temp_file()
    -- Random component: /tmp/aisubs_<time>_<rand>.txt — a predictable name
    -- lets a local attacker pre-plant a symlink that our open() would follow.
    local unique = tostring(os.time()) .. "_" .. tostring(math.random(100000, 999999))
    if is_windows() then
        local tmp = os.getenv("TEMP") or os.getenv("TMP") or (get_home() .. "\\AppData\\Local\\Temp")
        return tmp .. "\\aisubs_" .. unique .. ".txt"
    else
        local tmp = os.getenv("TMPDIR") or "/tmp"
        return tmp .. "/aisubs_" .. unique .. ".txt"
    end
end

-- POSIX sh single-quote escaping. Every interpolated value lands inside a
-- shell command (os.execute → /bin/sh -c); double quotes alone are NOT
-- sufficient — $(...) and backticks execute even inside them.
local function shq(s)
    return "'" .. string.gsub(s or "", "'", "'\\''") .. "'"
end

----------------------------------------------------------------
-- Remembered settings
----------------------------------------------------------------
-- vlc.config.set() only accepts existing vlcrc option names (it fails for
-- custom keys), so the dialog keeps a small key=value file in the extension's
-- user data dir instead. Values are whitelisted on load: they end up inside a
-- shell command (VSCL_AISUBS_BACKEND=..., model/language args).

function settings_dir()
    local base
    local ok, dir = pcall(function() return vlc.config.userdatadir() end)
    if ok and dir and dir ~= "" then
        base = dir
    else
        base = get_home() .. "/.local/share/vlc"
    end
    return base .. (is_windows() and "\\vlc-ai-subs" or "/vlc-ai-subs")
end

function settings_file()
    return settings_dir() .. (is_windows() and "\\settings.conf" or "/settings.conf")
end

function ensure_dir(path)
    if is_windows() then
        os.execute('mkdir "' .. path .. '" 2>nul')
    else
        os.execute("mkdir -p " .. shq(path) .. " 2>/dev/null")
    end
end

local function on_list(value, list)
    for _, name in ipairs(list) do
        if name == value then return name end
    end
    return nil
end

-- Language tag check (en, ja, yue, en-US, zh-CN ...), used both for the dialog
-- field and for values restored from disk — they land inside a shell command.
-- NOTE: Lua patterns cannot quantify a capture group, so the old
-- "^[a-zA-Z][a-zA-Z0-9]*(-[a-zA-Z0-9]+)*$" was parsed as "letters, digits, then
-- a literal *" and matched NOTHING: every explicit language code (en, ja, es…)
-- was rejected with "invalid language code". Validate segment by segment.
function is_language_code(value)
    if type(value) ~= "string" or value == "" or #value > 20 then return false end
    local segments = 0
    for segment in string.gmatch(value, "[^-]+") do
        segments = segments + 1
        if not string.match(segment, "^%a[%w]*$") then return false end
    end
    -- No empty segments: rejects "en-", "-en", "en--US".
    return segments == select(2, string.gsub(value, "%-", "")) + 1
end

local function names_of(pairs_table)
    local out = {}
    for _, it in ipairs(pairs_table) do out[#out + 1] = it[1] end
    return out
end

function load_settings()
    local out = {}
    local f = io.open(settings_file(), "r")
    if not f then return out end
    for line in f:lines() do
        local k, v = string.match(line, "^([%w_]+)=(.*)$")
        if k and on_list(k, SETTING_KEYS) then out[k] = v end
    end
    f:close()
    out.engine = on_list(out.engine, names_of(ENGINES))
    out.model  = on_list(out.model, names_of(MODELS))
    out.task   = on_list(out.task, names_of(TASKS))
    out.mode   = on_list(out.mode, names_of(MODES))
    if out.language and not is_language_code(out.language) then
        out.language = nil
    end
    return out
end

function save_settings(values)
    ensure_dir(settings_dir())
    local f = io.open(settings_file(), "w")
    if not f then return false end
    for _, k in ipairs(SETTING_KEYS) do
        f:write(k .. "=" .. tostring(values[k] or "") .. "\n")
    end
    f:close()
    return true
end

function pid_file_for(mirror)
    return (string.gsub(mirror or "", "%.txt$", ".pid"))
end

-- Stop the run started from tmp_file. The CLI records its PID next to the
-- mirror file and stops the ML child itself when signalled (VLC's Lua has no
-- process API); the PID is whitelisted to digits before it reaches the shell.
function cancel_run()
    local tmp = _poll_tmp
    if _poll_tmr then
        pcall(function() _poll_tmr:cancel() end)
        _poll_tmr = nil
    end
    if not tmp then
        set_status("Nothing is running.")
        return false
    end

    local killed = false
    local f = io.open(pid_file_for(tmp), "r")
    if f then
        local pid = string.match(f:read("*a") or "", "(%d+)")
        f:close()
        if pid then
            if is_windows() then
                os.execute("taskkill /F /T /PID " .. pid .. " 2>nul")
            else
                os.execute("kill " .. pid .. " 2>/dev/null")
            end
            killed = true
        end
    end

    pcall(function() os.remove(tmp) end)
    pcall(function() os.remove(string.gsub(tmp, "%.txt$", ".srt")) end)
    pcall(function() os.remove(pid_file_for(tmp)) end)
    _poll_tmp, _poll_cue, _poll_status, _poll_cues = nil, nil, nil, 0
    if progress_bar then progress_bar:set_value(0) end
    if cue_label then cue_label:set_text("") end
    if details_label then details_label:set_text("") end
    set_status(killed and "Cancelled." or "Nothing is running.")
    return killed
end

function get_media_duration()
    -- Try VLC player first (currently playing media).
    -- item:duration() already returns SECONDS (VLC 3.x + 4.x Lua README) —
    -- no /1000 here, or the ETA/progress estimate would be 1000x too fast.
    local item = get_input_item()
    if item then
        local dur = item:duration()
        if dur and dur > 0 then
            return dur
        end
    end
    return 0
end

----------------------------------------------------------------
-- Media path
----------------------------------------------------------------

function get_media_path()
    local item = get_input_item()
    if not item then return nil, "No media is currently playing." end
    local uri = item:uri()
    if not uri then return nil, "Cannot get media URI." end
    if not string.find(uri, "^file://") then return nil, "Only local files are supported." end

    -- Strip file:// prefix
    local path = string.gsub(uri, "^file://", "")

    -- URL-decode percent-encoded characters
    path = string.gsub(path, "%%(%x%x)", function(hex)
        return string.char(tonumber(hex, 16))
    end)

    -- On Windows, VLC produces file:///C:/path → after strip → /C:/path
    -- Remove the leading slash before the drive letter
    if is_windows() then
        path = string.gsub(path, "^/([A-Za-z]:)", "%1")
        path = string.gsub(path, "/", "\\")
    end

    vlc.msg.info("[AI Subs] media path: " .. path)
    return path, nil
end

----------------------------------------------------------------
-- Locate the Python backend script
----------------------------------------------------------------

function find_script()
    local home = get_home()
    local candidates = {}

    if is_windows() then
        local appdata = os.getenv("APPDATA") or (home .. "\\AppData\\Roaming")
        table.insert(candidates, home .. "\\Documents\\vlc-ai-subs\\aisubs_whisper.py")
        table.insert(candidates, home .. "\\Desktop\\vlc-ai-subs\\aisubs_whisper.py")
        table.insert(candidates, home .. "\\Desktop\\aisubs\\aisubs_whisper.py")
        table.insert(candidates, home .. "\\vlc-ai-subs\\aisubs_whisper.py")
        table.insert(candidates, appdata .. "\\vlc-ai-subs\\aisubs_whisper.py")
        table.insert(candidates, "C:\\vlc-ai-subs\\aisubs_whisper.py")
    else
        table.insert(candidates, home .. "/Desktop/vlc-ai-subs/aisubs_whisper.py")
        table.insert(candidates, home .. "/Desktop/aisubs/aisubs_whisper.py")
        table.insert(candidates, home .. "/vlc-ai-subs/aisubs_whisper.py")
        table.insert(candidates, home .. "/.local/share/vlc-ai-subs/aisubs_whisper.py")
        table.insert(candidates, "/opt/vlc-ai-subs/aisubs_whisper.py")
        table.insert(candidates, "/usr/local/share/vlc-ai-subs/aisubs_whisper.py")
    end

    for _, path in ipairs(candidates) do
        local f = io.open(path, "r")
        if f then f:close(); return path end
    end
    return nil
end

function find_python(script_dir)
    local sep = is_windows() and "\\" or "/"

    -- venv on Unix
    local p = script_dir .. sep .. "venv" .. sep .. "bin" .. sep .. "python3"
    local f = io.open(p, "r")
    if f then f:close(); return p end

    -- venv on Windows
    p = script_dir .. sep .. "venv" .. sep .. "Scripts" .. sep .. "python.exe"
    f = io.open(p, "r")
    if f then f:close(); return p end

    return is_windows() and "python" or "python3"
end

----------------------------------------------------------------
-- Main entry
----------------------------------------------------------------

function start_generation()
    -- A new run supersedes the previous one: stop it properly (this used to
    -- only stop polling and leave the old process transcribing).
    if _poll_tmr then cancel_run() end

    local media_path, err = get_media_path()
    if not media_path then
        set_status("Error: " .. err)
        return
    end

    local script = find_script()
    if not script then
        set_status("Error: aisubs_whisper.py not found. Run setup.sh first.")
        return
    end

    local script_dir = string.match(script, "(.+)[/\\][^/\\]+$") or "."
    local python    = find_python(script_dir)
    local model     = get_model_name()
    local language  = lang_input:get_text() or "auto"
    -- Whitelist language codes: this free-text field lands inside a shell
    -- command below, so reject anything that is not a lang tag (en, zh-CN…).
    if language ~= "auto" and not is_language_code(language) then
        set_status("Error: invalid language code: " .. language)
        return
    end
    local task      = get_task()
    local mode      = get_mode()
    local tmp_file  = get_temp_file()

    -- Write sentinel so we can detect if Python started writing.
    -- Prefer exclusive create ("wx" fails on a pre-planted symlink instead of
    -- following it); older Lua builds fall back to "w" — the random temp name
    -- already blocks the symlink race regardless.
    local ok, test_f = pcall(io.open, tmp_file, "wx")
    if not ok or not test_f then
        test_f = io.open(tmp_file, "w")
    end
    if not test_f then
        set_status("Error: cannot write to temp dir: " .. tmp_file)
        return
    end
    test_f:write("init\n")
    test_f:close()

    -- Build and launch command NON-BLOCKING so VLC's thread is not frozen.
    -- Windows: VBScript with bWaitOnReturn=False → wscript exits immediately.
    -- Unix:    trailing & → shell forks Python and exits immediately.
    -- In both cases io.popen returns at once and we poll tmp_file via vlc.timer.
    local engine_choice = get_engine()
    -- Auto/Parakeet resolve to an engine that can actually do this run (Parakeet
    -- has no translation and no detection); engine_note explains a substitution.
    local engine, engine_note = engine_for(engine_choice, language, task)
    -- Parakeet ignores the model dropdown (fixed parakeet-tdt-0.6b-v2);
    -- show the model that actually runs in the status lines.
    local shown_model = (engine == "parakeet") and "parakeet-tdt-0.6b-v2" or model
    -- Remember the dialog choices (not the substitution) for the next session.
    save_settings({
        engine = engine_choice, model = model, language = language,
        task = task, mode = mode,
    })
    -- Realtime-OSD mode: write the SRT to a writable temp path (never next
    -- to the media, and immune to read-only media dirs). "Generate & Load"
    -- passes "" so the caller derives <media>.srt.
    local srt_arg = ""
    if mode == "realtime" then
        srt_arg = string.gsub(tmp_file, "%.txt$", ".srt")
    end
    local cmd
    if is_windows() then
        local vbs_file = string.gsub(tmp_file, "%.txt$", ".vbs")
        local vf = io.open(vbs_file, "w")
        if not vf then
            set_status("Error: cannot write helper file: " .. vbs_file)
            return
        end
        -- In VBScript string literals a literal double-quote is written as ""
        local raw_cmd = string.format('"%s" -u "%s" "%s" "%s" "%s" "%s" "%s" "%s" --debug',
            python, script, media_path, model, language, task, tmp_file, srt_arg)
        local vbs_cmd = raw_cmd:gsub('"', '""'):gsub("[\r\n]", "")  -- + strip line breaks (VBS line injection)
        vf:write('Set sh = CreateObject("WScript.Shell")\n')
        if engine ~= "" then
            -- Windows can't prefix env vars on the command line; set them on
            -- the child process via WScript.Shell's environment instead.
            vf:write('sh.Environment("PROCESS")("VSCL_AISUBS_BACKEND") = "' .. engine .. '"\n')
        end
        vf:write('sh.Run "' .. vbs_cmd .. '", 0, False\n')  -- 0=hidden, False=don't wait
        vf:close()
        cmd = 'wscript.exe /nologo "' .. vbs_file .. '"'
    else
        local env_prefix = ""
        if engine ~= "" then
            env_prefix = "VSCL_AISUBS_BACKEND=" .. engine .. " "
        end
        cmd = string.format('%s%s -u %s %s %s %s %s %s %s --debug',
            env_prefix, shq(python), shq(script), shq(media_path), shq(model),
            shq(language), shq(task), shq(tmp_file), shq(srt_arg))
    end

    vlc.msg.info("[AI Subs] python: " .. python)
    vlc.msg.info("[AI Subs] media:  " .. media_path)
    vlc.msg.info("[AI Subs] tmp:    " .. tmp_file)

    -- Launch via os.execute with & for true non-blocking background.
    -- io.popen blocks in VLC's Lua sandbox; os.execute returns instantly.
    if is_windows() then
        local ok = os.execute(cmd)
        if ok ~= 0 then
            set_status("Error: failed to launch Python. Check VLC logs.")
            return
        end
    else
        os.execute(cmd .. " &")
    end

    -- Poll tmp_file every second; VLC's thread stays free the whole time
    _poll_tmp    = tmp_file
    _poll_mode   = mode
    _poll_model  = shown_model
    _poll_engine = engine_label(engine)
    _poll_secs   = 0
    _poll_shown  = 0
    _poll_status = nil
    _poll_cue    = nil
    _poll_cues   = 0
    if details_label then
        details_label:set_text(string.format("%s · %s · %s · %s%s",
            _poll_engine, shown_model, language, task,
            engine_note and (" — " .. engine_note) or ""))
    end
    if cue_label then cue_label:set_text("") end

    -- Estimate total time: rough RTF × audio duration (engine-dependent).
    -- Parakeet ≈ 10× realtime on CPU (0.1); WhisperX ≈ 0.3× GPU / 2× CPU.
    local duration = get_media_duration()
    _poll_duration = duration or 0
    if _poll_duration > 0 then
        if engine == "parakeet" then
            _poll_est_total = _poll_duration * 0.1
        else
            _poll_est_total = _poll_duration * 0.5
        end
    else
        _poll_est_total = 30  -- unknown, guess 30s
    end

    set_status("Transcribing with " .. _poll_engine .. " (" .. shown_model .. ")... please wait")
    progress_bar:set_value(0)

    -- Show debug command so user can run it from terminal if needed
    debug_label:set_text("Debug: " .. cmd)
    _poll_tmr = vlc.timer(poll_progress)
    _poll_tmr:schedule(POLL_US)
end

----------------------------------------------------------------
-- Polling callback — called by vlc.timer every POLL_US microseconds
----------------------------------------------------------------

function poll_progress()
    _poll_secs = _poll_secs + (POLL_US / 1000000)

    -- Update progress bar based on elapsed vs estimated
    if _poll_est_total > 0 then
        local pct = math.min(95, (_poll_secs / _poll_est_total) * 100)
        progress_bar:set_value(pct)
    end

    local f = io.open(_poll_tmp, "r")
    if not f then
        -- Temp file gone — shouldn't happen; keep waiting
        local eta = math.max(0, _poll_est_total - _poll_secs)
        set_status(string.format("Transcribing with %s (%s)... %ds  ETA ~%ds", _poll_engine, _poll_model, _poll_secs, eta))
        _poll_tmr:schedule(POLL_US)
        return
    end

    -- Walk every line: 'sub' lines feed the OSD as they are produced
    -- (real-time mode), while the LAST line decides whether the run finished.
    -- Only cues past the high-water mark are pushed, so a poll that sees the
    -- same file twice never duplicates a caption.
    local last_line, seen = nil, 0
    for line in f:lines() do
        last_line = line
        if string.find(line, '"type"', 1, true) then
            local ev = parse_json(line)
            if ev and ev.type == "sub" then
                seen = seen + 1
                _poll_cue = ev.text
                if _poll_mode == "realtime" and seen > _poll_shown then
                    show_cue_osd(ev)
                end
            elseif ev and ev.type == "status" then
                -- The CLI's own phase lines ("Backend: …", "Transcribing…")
                _poll_status = ev.msg
            end
        end
    end
    f:close()
    if seen > _poll_shown then _poll_shown = seen end
    if seen > _poll_cues then _poll_cues = seen end

    if not last_line or last_line == "init" then
        -- Python hasn't written output yet
        set_status(string.format("Loading model / starting... %ds", _poll_secs))
        update_details()
        _poll_tmr:schedule(POLL_US)
        return
    end

    local d = parse_json(last_line)
    if d and (d.type == "done" or d.type == "error") then
        -- Python finished — process results
        _poll_tmr = nil
        progress_bar:set_value(100)
        update_details()
        process_results(_poll_tmp, _poll_mode)
    else
        local eta = math.max(0, _poll_est_total - _poll_secs)
        set_status(string.format("Transcribing with %s (%s)... %ds  ETA ~%ds", _poll_engine, _poll_model, _poll_secs, eta))
        update_details()
        _poll_tmr:schedule(POLL_US)
    end
end

-- Keep the details/cue labels in step with the JSONL stream (phase lines from
-- the CLI plus the newest cue) without touching the dialog per event.
function update_details()
    if details_label then
        local parts = {
            _poll_engine or "?", _poll_model or "?",
            string.format("%ds", _poll_secs),
            string.format("ETA ~%ds", math.max(0, _poll_est_total - _poll_secs)),
        }
        if _poll_cues > 0 then
            parts[#parts + 1] = string.format("%d cues", _poll_cues)
        end
        local text = table.concat(parts, " · ")
        if _poll_status then text = text .. " · " .. _poll_status end
        details_label:set_text(text)
    end
    if cue_label and _poll_cue then
        -- Cues carry SRT line breaks; the dialog shows one line.
        cue_label:set_text("Last cue: " .. string.gsub(_poll_cue, "%s+", " "))
    end
end

----------------------------------------------------------------
-- Process results from temp file
----------------------------------------------------------------

function process_results(tmp_file, mode)
    local f = io.open(tmp_file, "r")
    if not f then
        set_status("Error: Whisper produced no output. Check VLC logs.")
        return
    end

    local srt_path  = nil
    local seg_count = 0

    for line in f:lines() do
        local d = parse_json(line)
        if d then
            if d.type == "error" then
                set_status("Error: " .. (d.msg or "unknown"))
                f:close()
                pcall(function() os.remove(tmp_file) end)
                return
            elseif d.type == "sub" then
                -- OSD is fed live by poll_progress; replaying the cues here
                -- would show every caption a second time.
                seg_count = seg_count + 1
            elseif d.type == "status" then
                set_status(d.msg or "")
            elseif d.type == "done" then
                srt_path  = d.srt_path
                seg_count = d.segments or seg_count
            end
        end
    end
    f:close()
    pcall(function() os.remove(tmp_file) end)

    if not srt_path then
        if seg_count == 0 then
            set_status("No speech detected — nothing to transcribe.")
        else
            set_status("Error: transcription failed. Check VLC logs for details.")
        end
        return
    end

    if mode == "srt" then
        load_subtitle(srt_path)
        set_status("Done! " .. seg_count .. " segments. Subtitles loaded.")
    else
        set_status("Done! " .. seg_count .. " segments. SRT: " .. srt_path)
    end
end

----------------------------------------------------------------
-- Helpers
----------------------------------------------------------------

function load_subtitle(srt_path)
    local f = io.open(srt_path, "r")
    if not f then return end
    f:close()
    if add_subtitle_track(srt_path) then
        vlc.msg.info("[AI Subs] Loaded: " .. srt_path)
    else
        vlc.msg.warn("[AI Subs] Auto-load failed. Add manually: " .. srt_path)
    end
end

function set_status(text)
    if status_label then status_label:set_text(text) end
    if dlg then dlg:update() end
end

function parse_json(str)
    if not str then return nil end
    local j = string.match(str, "%b{}")
    if not j then return nil end
    local r = {}
    for k, v in string.gmatch(j, '"([^"]+)"%s*:%s*"(.-)"') do
        v = string.gsub(v, "\\n", "\n")
        v = string.gsub(v, "\\t", "\t")
        v = string.gsub(v, '\\"', '"')
        v = string.gsub(v, "\\\\", "\\")
        r[k] = v
    end
    for k, v in string.gmatch(j, '"([^"]+)"%s*:%s*([%d%.%-]+)') do
        if not r[k] then r[k] = tonumber(v) end
    end
    return r
end
