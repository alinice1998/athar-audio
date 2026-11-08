/**
 * Quranic Audio Style & Reverb Cloner - Frontend Client
 * Handles WaveSurfer, Live Recording, DSP Parameter Controls,
 * A/B Real-time Comparison, and Chart.js Spectrum Rendering.
 */

// Global State
const state = {
    refFile: null,
    refCachedHash: null,
    targetFile: null,
    recordedBlob: null,
    isRecording: false,
    mediaRecorder: null,
    audioContext: null,
    analyser: null,
    meterInterval: null,
    recordTimerInterval: null,
    recordSeconds: 0,
    activeComparison: 'master', // 'master', 'target', 'ref'
    waves: {
        ref: null,
        target: null,
        master: null
    },
    audioUrls: {},
    spectrumChart: null,
    componentAudios: {
        refDry: null,
        refReverb: null
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initWaveSurfers();
    initSystemStatus();
    initEventListeners();
    initSliders();
});

// -------------------------------------------------------------
// 1. Initialize WaveSurfer Players
// -------------------------------------------------------------
function initWaveSurfers() {
    // Reference WaveSurfer
    state.waves.ref = WaveSurfer.create({
        container: '#refWaveform',
        waveColor: '#475569',
        progressColor: '#d4af37',
        cursorColor: '#facc15',
        barWidth: 2,
        barRadius: 2,
        height: 64,
        normalize: true
    });

    state.waves.ref.on('audioprocess', (time) => {
        const total = state.waves.ref.getDuration();
        document.getElementById('refTimeDisplay').textContent = `${formatTime(time)} / ${formatTime(total)}`;
    });

    state.waves.ref.on('ready', () => {
        const total = state.waves.ref.getDuration();
        document.getElementById('refTimeDisplay').textContent = `0:00 / ${formatTime(total)}`;
    });

    // Target User WaveSurfer
    state.waves.target = WaveSurfer.create({
        container: '#targetWaveform',
        waveColor: '#475569',
        progressColor: '#10b981',
        cursorColor: '#34d399',
        barWidth: 2,
        barRadius: 2,
        height: 64,
        normalize: true
    });

    state.waves.target.on('audioprocess', (time) => {
        const total = state.waves.target.getDuration();
        document.getElementById('targetTimeDisplay').textContent = `${formatTime(time)} / ${formatTime(total)}`;
    });

    state.waves.target.on('ready', () => {
        const total = state.waves.target.getDuration();
        document.getElementById('targetTimeDisplay').textContent = `0:00 / ${formatTime(total)}`;
    });

    // Master Output WaveSurfer (A/B player)
    state.waves.master = WaveSurfer.create({
        container: '#masterWaveform',
        waveColor: '#334155',
        progressColor: '#10b981',
        cursorColor: '#38bdf8',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        normalize: true
    });

    state.waves.master.on('audioprocess', (time) => {
        const total = state.waves.master.getDuration();
        document.getElementById('masterTimeDisplay').textContent = `${formatTime(time)} / ${formatTime(total)}`;
    });

    state.waves.master.on('ready', () => {
        const total = state.waves.master.getDuration();
        document.getElementById('masterTimeDisplay').textContent = `0:00 / ${formatTime(total)}`;
    });

    state.waves.master.on('finish', () => {
        document.getElementById('mainPlayPauseBtn').innerHTML = '<i class="fa-solid fa-play"></i>';
    });
}

// -------------------------------------------------------------
// 2. System Status (GPU/CPU Check)
// -------------------------------------------------------------
async function initSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const badge = document.getElementById('gpuStatusText');
        if (data.cuda_available) {
            badge.textContent = `تسريع كرت الشاشة: ${data.gpu_name} (${data.vram_gb} GB VRAM)`;
        } else {
            badge.textContent = `المعالج: ${data.gpu_name}`;
        }
    } catch (e) {
        document.getElementById('gpuStatusText').textContent = 'النظام جاهز محلياً';
    }
}

// -------------------------------------------------------------
// 3. UI Controls & Event Listeners
// -------------------------------------------------------------
function initEventListeners() {
    // Reference File Upload
    const refDropzone = document.getElementById('refDropzone');
    const refFileInput = document.getElementById('refFileInput');
    refDropzone.addEventListener('click', () => refFileInput.click());
    refFileInput.addEventListener('change', (e) => handleRefFile(e.target.files[0]));
    setupDragDrop(refDropzone, handleRefFile);

    // Target File Upload
    const targetDropzone = document.getElementById('targetDropzone');
    const targetFileInput = document.getElementById('targetFileInput');
    targetDropzone.addEventListener('click', () => targetFileInput.click());
    targetFileInput.addEventListener('change', (e) => handleTargetFile(e.target.files[0]));
    setupDragDrop(targetDropzone, handleTargetFile);

    // Microphone Recording
    document.getElementById('recordBtn').addEventListener('click', toggleRecording);

    // Playback Buttons
    document.getElementById('refPlayBtn').addEventListener('click', () => {
        state.waves.ref.playPause();
    });

    document.getElementById('targetPlayBtn').addEventListener('click', () => {
        state.waves.target.playPause();
    });

    document.getElementById('mainPlayPauseBtn').addEventListener('click', toggleMainPlayPause);

    // A/B Comparison Switching
    document.getElementById('switchMasterBtn').addEventListener('click', () => switchComparisonTrack('master'));
    document.getElementById('switchTargetBtn').addEventListener('click', () => switchComparisonTrack('target'));
    document.getElementById('switchRefBtn').addEventListener('click', () => switchComparisonTrack('ref'));

    // Component Audios (Ref Dry & Reverb)
    document.getElementById('playRefDryBtn').addEventListener('click', () => toggleComponentAudio('refDry'));
    document.getElementById('playRefReverbBtn').addEventListener('click', () => toggleComponentAudio('refReverb'));

    // Process Button
    document.getElementById('processBtn').addEventListener('click', startPipelineProcessing);

    // Reset Parameters
    document.getElementById('resetParamsBtn').addEventListener('click', resetParameters);

    // Quick Sample Buttons Loader
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const type = btn.dataset.type; // 'ref' or 'target'
            const filename = btn.dataset.file;
            const url = `/audio/samples/${filename}`;

            btn.classList.add('animate-pulse');
            try {
                const res = await fetch(url);
                const blob = await res.blob();
                const file = new File([blob], filename, { type: blob.type || 'audio/mpeg' });

                if (type === 'ref') {
                    await handleRefFile(file);
                } else if (type === 'target') {
                    handleTargetFile(file);
                }
            } catch (err) {
                console.error('Failed to load sample:', err);
            } finally {
                btn.classList.remove('animate-pulse');
            }
        });
    });
}

function initSliders() {
    const eqSlider = document.getElementById('eqIntensitySlider');
    const eqVal = document.getElementById('eqIntensityVal');
    eqSlider.addEventListener('input', () => {
        eqVal.textContent = `${eqSlider.value}%`;
    });

    const revSlider = document.getElementById('reverbMixSlider');
    const revVal = document.getElementById('reverbMixVal');
    revSlider.addEventListener('input', () => {
        revVal.textContent = `${revSlider.value}% (يدوي)`;
    });

    const decaySlider = document.getElementById('decayScaleSlider');
    const decayVal = document.getElementById('decayScaleVal');
    decaySlider.addEventListener('input', () => {
        decayVal.textContent = `${(decaySlider.value / 100).toFixed(1)}x`;
    });
}

function resetParameters() {
    document.getElementById('eqIntensitySlider').value = 80;
    document.getElementById('eqIntensityVal').textContent = '80%';
    document.getElementById('reverbMixSlider').value = 35;
    document.getElementById('reverbMixVal').textContent = 'تلقائي (مكتشف)';
    document.getElementById('decayScaleSlider').value = 100;
    document.getElementById('decayScaleVal').textContent = '1.0x';
    document.getElementById('duckingToggle').checked = true;
    document.getElementById('cleanToggle').checked = true;
    document.getElementById('eqEngineSelect').value = 'spectral';
    document.getElementById('reverbModeSelect').value = 'parametric';
}

// -------------------------------------------------------------
// 4. File Drag & Drop Handlers
// -------------------------------------------------------------
function setupDragDrop(element, callback) {
    element.addEventListener('dragover', (e) => {
        e.preventDefault();
        element.classList.add('border-emerald-500', 'bg-slate-900/80');
    });
    element.addEventListener('dragleave', () => {
        element.classList.remove('border-emerald-500', 'bg-slate-900/80');
    });
    element.addEventListener('drop', (e) => {
        e.preventDefault();
        element.classList.remove('border-emerald-500', 'bg-slate-900/80');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            callback(e.dataTransfer.files[0]);
        }
    });
}

async function handleRefFile(file) {
    if (!file) return;
    state.refFile = file;
    document.getElementById('refFileLabel').textContent = file.name;
    document.getElementById('refFileNameDisplay').textContent = file.name;
    document.getElementById('refWaveContainer').classList.remove('hidden');

    const url = URL.createObjectURL(file);
    state.waves.ref.load(url);

    // Check caching status
    try {
        const formData = new FormData();
        formData.append('ref_file', file);
        const res = await fetch('/api/check-cache', { method: 'POST', body: formData });
        const cacheData = await res.json();
        const badge = document.getElementById('refCacheBadge');
        if (cacheData.is_cached) {
            state.refCachedHash = cacheData.hash;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    } catch (e) {
        console.error(e);
    }
}

function handleTargetFile(file) {
    if (!file) return;
    state.targetFile = file;
    state.recordedBlob = null;
    document.getElementById('targetFileLabel').textContent = file.name;
    document.getElementById('targetFileNameDisplay').textContent = file.name;
    document.getElementById('targetWaveContainer').classList.remove('hidden');

    const url = URL.createObjectURL(file);
    state.waves.target.load(url);
}

// -------------------------------------------------------------
// 5. Microphone Recording & Audio Metering
// -------------------------------------------------------------
async function toggleRecording() {
    const recordBtn = document.getElementById('recordBtn');
    const statusText = document.getElementById('recordStatusText');
    const timerText = document.getElementById('recordTimer');
    const meterBar = document.getElementById('micMeterBar');

    if (!state.isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = state.audioContext.createMediaStreamSource(stream);
            state.analyser = state.audioContext.createAnalyser();
            state.analyser.fftSize = 256;
            source.connect(state.analyser);

            const chunks = [];
            state.mediaRecorder = new MediaRecorder(stream);
            state.mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            state.mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                state.recordedBlob = blob;
                state.targetFile = new File([blob], 'user_mic_recording.wav', { type: 'audio/wav' });
                
                document.getElementById('targetFileLabel').textContent = 'تسجيل ميكروفون مباشر (جاهز)';
                document.getElementById('targetFileNameDisplay').textContent = 'user_mic_recording.wav';
                document.getElementById('targetWaveContainer').classList.remove('hidden');
                
                const url = URL.createObjectURL(blob);
                state.waves.target.load(url);

                // Stop tracks
                stream.getTracks().forEach(t => t.stop());
            };

            state.mediaRecorder.start();
            state.isRecording = true;
            recordBtn.classList.replace('bg-red-600', 'bg-amber-500');
            recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
            statusText.textContent = 'جاري التسجيل... انقر للإيقاف';

            // Timer
            state.recordSeconds = 0;
            state.recordTimerInterval = setInterval(() => {
                state.recordSeconds++;
                const mins = String(Math.floor(state.recordSeconds / 60)).padStart(2, '0');
                const secs = String(state.recordSeconds % 60).padStart(2, '0');
                timerText.textContent = `${mins}:${secs}`;
            }, 1000);

            // Meter Loop
            const pcmData = new Uint8Array(state.analyser.frequencyBinCount);
            state.meterInterval = setInterval(() => {
                state.analyser.getByteFrequencyData(pcmData);
                let sum = 0;
                for (let i = 0; i < pcmData.length; i++) sum += pcmData[i];
                const avg = sum / pcmData.length;
                const pct = Math.min(100, Math.round((avg / 128) * 100));
                meterBar.style.width = `${pct}%`;
            }, 50);

        } catch (err) {
            alert('تعذر الوصول إلى الميكروفون: ' + err.message);
        }
    } else {
        // Stop recording
        state.mediaRecorder.stop();
        state.isRecording = false;
        clearInterval(state.meterInterval);
        clearInterval(state.recordTimerInterval);
        meterBar.style.width = '0%';
        recordBtn.classList.replace('bg-amber-500', 'bg-red-600');
        recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
        statusText.textContent = 'تم تسجيل الصوت بنجاح';
    }
}

// -------------------------------------------------------------
// 6. Pipeline Execution & Real-Time Progress
// -------------------------------------------------------------
async function startPipelineProcessing() {
    if (!state.refFile && !state.refCachedHash) {
        alert('يرجى اختيار أو رفع ملف التلاوة المرجعية أولاً.');
        return;
    }
    if (!state.targetFile && !state.recordedBlob) {
        alert('يرجى رفع تلاوتك الخام أو التسجيل عبر الميكروفون أولاً.');
        return;
    }

    const processBtn = document.getElementById('processBtn');
    const progressSection = document.getElementById('progressSection');
    const outputSection = document.getElementById('outputSection');
    const progressBar = document.getElementById('progressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const currentStageText = document.getElementById('currentStageText');

    processBtn.disabled = true;
    processBtn.classList.add('opacity-50', 'cursor-not-allowed');
    progressSection.classList.remove('hidden');
    outputSection.classList.add('hidden');

    // Simulate animated progress steps while backend completes
    let fakePct = 10;
    progressBar.style.width = '10%';
    progressPercentage.textContent = '10%';
    highlightStep(1);

    const progressTimer = setInterval(() => {
        if (fakePct < 90) {
            fakePct += 5;
            progressBar.style.width = `${fakePct}%`;
            progressPercentage.textContent = `${fakePct}%`;
            if (fakePct >= 30) highlightStep(2);
            if (fakePct >= 55) highlightStep(3);
            if (fakePct >= 75) highlightStep(4);
            if (fakePct >= 88) highlightStep(5);
        }
    }, 400);

    const formData = new FormData();
    if (state.refFile) formData.append('ref_file', state.refFile);
    if (state.refCachedHash) formData.append('ref_cached_hash', state.refCachedHash);

    const targetToUpload = state.recordedBlob 
        ? new File([state.recordedBlob], 'user_recording.wav', { type: 'audio/wav' })
        : state.targetFile;
    formData.append('target_file', targetToUpload);

    const eqVal = parseFloat(document.getElementById('eqIntensitySlider').value) / 100.0;
    formData.append('eq_intensity', eqVal);

    const revInputVal = document.getElementById('reverbMixSlider').value;
    const isAutoRev = document.getElementById('reverbMixVal').textContent.includes('تلقائي');
    if (!isAutoRev) {
        formData.append('reverb_mix', parseFloat(revInputVal) / 100.0);
    }

    const decayVal = parseFloat(document.getElementById('decayScaleSlider').value) / 100.0;
    formData.append('decay_scale', decayVal);
    formData.append('enable_ducking', document.getElementById('duckingToggle').checked);
    formData.append('clean_target', document.getElementById('cleanToggle').checked);
    formData.append('eq_engine', document.getElementById('eqEngineSelect').value);
    formData.append('reverb_mode', document.getElementById('reverbModeSelect').value);

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });

        clearInterval(progressTimer);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'فشلت معالجة الصوت.');
        }

        const data = await response.json();
        
        // 100% complete
        progressBar.style.width = '100%';
        progressPercentage.textContent = '100%';
        currentStageText.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i> اكتملت المعالجة بنجاح!';
        highlightStep(5);

        setTimeout(() => {
            progressSection.classList.add('hidden');
            renderOutputStudio(data);
        }, 600);

    } catch (err) {
        clearInterval(progressTimer);
        alert('حدث خطأ أثناء المعالجة: ' + err.message);
        progressSection.classList.add('hidden');
    } finally {
        processBtn.disabled = false;
        processBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
}

function highlightStep(stepNum) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`step-${i}`);
        if (i <= stepNum) {
            el.className = 'text-emerald-400 font-bold';
        } else {
            el.className = 'text-slate-500';
        }
    }
}

// -------------------------------------------------------------
// 7. Render Output Studio & Synchronized A/B Switching
// -------------------------------------------------------------
function renderOutputStudio(data) {
    state.audioUrls = data.urls;
    const outputSection = document.getElementById('outputSection');
    outputSection.classList.remove('hidden');

    // Badges & Metrics
    document.getElementById('elapsedTimeBadge').textContent = `في ${data.elapsed_seconds} ثانية`;
    document.getElementById('statRt60').textContent = `${data.acoustic_analysis.rt60_seconds} s`;
    document.getElementById('statPredelay').textContent = `${data.acoustic_analysis.predelay_ms} ms`;
    document.getElementById('statMix').textContent = `${Math.round(data.parameters.reverb_mix * 100)}%`;
    document.getElementById('statPeak').textContent = `${data.mastering_stats.peak_db} dBFS`;

    // Download Links
    document.getElementById('downloadMasterBtn').href = data.urls.final_master;
    document.getElementById('downloadIrBtn').href = data.urls.extracted_ir;

    // Load Master Track initially into WaveSurfer
    switchComparisonTrack('master');

    // Prepare Component Audio elements
    state.componentAudios.refDry = new Audio(data.urls.ref_dry);
    state.componentAudios.refReverb = new Audio(data.urls.ref_reverb);

    // Render Spectrum Chart
    renderSpectrumChart(data.spectrum_data);

    // Smooth scroll to output
    outputSection.scrollIntoView({ behavior: 'smooth' });
}

function switchComparisonTrack(trackKey) {
    const currentTime = state.waves.master.getCurrentTime();
    const wasPlaying = state.waves.master.isPlaying();

    state.activeComparison = trackKey;
    const btnMaster = document.getElementById('switchMasterBtn');
    const btnTarget = document.getElementById('switchTargetBtn');
    const btnRef = document.getElementById('switchRefBtn');
    const label = document.getElementById('activeTrackLabel');

    // Reset button styles
    [btnMaster, btnTarget, btnRef].forEach(b => {
        b.className = 'px-4 py-2 rounded-xl text-xs font-bold transition text-slate-400 hover:text-white';
    });

    let targetUrl = state.audioUrls.final_master;
    if (trackKey === 'master') {
        btnMaster.className = 'px-4 py-2 rounded-xl text-xs font-bold transition bg-emerald-500 text-white shadow-md';
        label.textContent = 'تلاوتك بالهندسة والماسترينغ (Mastered Cloned)';
        label.className = 'text-xs font-bold text-emerald-400 block';
        state.waves.master.setOptions({ progressColor: '#10b981' });
        targetUrl = state.audioUrls.final_master;
    } else if (trackKey === 'target') {
        btnTarget.className = 'px-4 py-2 rounded-xl text-xs font-bold transition bg-emerald-500 text-white shadow-md';
        label.textContent = 'تلاوتك الخام الأصلية بدون صدى (Dry Voice)';
        label.className = 'text-xs font-bold text-slate-300 block';
        state.waves.master.setOptions({ progressColor: '#94a3b8' });
        targetUrl = state.audioUrls.target_clean || state.audioUrls.target_raw;
    } else if (trackKey === 'ref') {
        btnRef.className = 'px-4 py-2 rounded-xl text-xs font-bold transition bg-gold-warm text-slate-900 shadow-md';
        label.textContent = 'التلاوة المرجعية الأصلية (Reference)';
        label.className = 'text-xs font-bold text-gold-warm block';
        state.waves.master.setOptions({ progressColor: '#d4af37' });
        targetUrl = state.audioUrls.reference;
    }

    if (targetUrl) {
        state.waves.master.load(targetUrl);
        state.waves.master.once('ready', () => {
            if (currentTime > 0) {
                state.waves.master.setTime(currentTime);
            }
            if (wasPlaying) {
                state.waves.master.play();
                document.getElementById('mainPlayPauseBtn').innerHTML = '<i class="fa-solid fa-pause"></i>';
            }
        });
    }
}

function toggleMainPlayPause() {
    state.waves.master.playPause();
    const isPlaying = state.waves.master.isPlaying();
    document.getElementById('mainPlayPauseBtn').innerHTML = isPlaying 
        ? '<i class="fa-solid fa-pause"></i>' 
        : '<i class="fa-solid fa-play"></i>';
}

function toggleComponentAudio(type) {
    const audio = state.componentAudios[type];
    const btn = type === 'refDry' ? document.getElementById('playRefDryBtn') : document.getElementById('playRefReverbBtn');
    if (!audio) return;

    if (audio.paused) {
        // Stop any other component audio
        if (state.componentAudios.refDry) state.componentAudios.refDry.pause();
        if (state.componentAudios.refReverb) state.componentAudios.refReverb.pause();
        document.getElementById('playRefDryBtn').innerHTML = '<i class="fa-solid fa-play"></i> استماع';
        document.getElementById('playRefReverbBtn').innerHTML = '<i class="fa-solid fa-play"></i> استماع';

        audio.play();
        btn.innerHTML = '<i class="fa-solid fa-pause"></i> إيقاف';
        audio.onended = () => {
            btn.innerHTML = '<i class="fa-solid fa-play"></i> استماع';
        };
    } else {
        audio.pause();
        btn.innerHTML = '<i class="fa-solid fa-play"></i> استماع';
    }
}

// -------------------------------------------------------------
// 8. Chart.js Frequency Spectrum Visualization
// -------------------------------------------------------------
function renderSpectrumChart(spectrumData) {
    const ctx = document.getElementById('spectrumChart').getContext('2d');
    if (state.spectrumChart) {
        state.spectrumChart.destroy();
    }

    state.spectrumChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: spectrumData.bands,
            datasets: [
                {
                    label: 'التلاوة المرجعية الجافة (Reference Dry)',
                    data: spectrumData.ref_dry,
                    borderColor: '#d4af37',
                    backgroundColor: 'rgba(212, 175, 55, 0.1)',
                    borderWidth: 2.5,
                    tension: 0.4,
                    pointRadius: 2
                },
                {
                    label: 'تلاوتك الخام (Target Dry)',
                    data: spectrumData.target_dry,
                    borderColor: '#94a3b8',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 2
                },
                {
                    label: 'تلاوتك بعد المطابقة والماسترينغ (Cloned)',
                    data: spectrumData.target_matched,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    borderWidth: 3,
                    tension: 0.4,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Cairo', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Tajawal', size: 10 } },
                    title: { display: true, text: 'القدرة (dBFS)', color: '#64748b' }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    rtl: true,
                    labels: {
                        color: '#cbd5e1',
                        font: { family: 'Cairo', size: 12 },
                        boxWidth: 14
                    }
                },
                tooltip: {
                    rtl: true,
                    titleFont: { family: 'Cairo' },
                    bodyFont: { family: 'Tajawal' }
                }
            }
        }
    });
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
}
