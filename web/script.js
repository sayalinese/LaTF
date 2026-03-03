document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = (window.__API_BASE__ || "http://127.0.0.1:5000").replace(/\/$/, "");
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const removeBtn = document.getElementById("remove-btn");
    const predictBtn = document.getElementById("predict-btn");
    const resultContainer = document.getElementById("result-container");
    const loader = document.getElementById("loader");
    const resultContent = document.getElementById("result-content");
    
    const statusBadge = document.getElementById("status-badge");
    const statusIcon = document.getElementById("status-icon");
    const resultText = document.getElementById("result-text");
    const meterFill = document.getElementById("meter-fill");
    const probReal = document.getElementById("prob-real");
    const probFake = document.getElementById("prob-fake");
    const probabilityText = document.getElementById("probability-text");

    // Debug panel
    const debugPanel = document.getElementById("debug-panel");
    const debugPre = document.getElementById("debug-pre");
    const debugToggle = document.getElementById("debug-toggle");
    
    // Heatmap Elements
    const heatmapContainer = document.getElementById("heatmap-container");
    const heatmapBg = document.getElementById("heatmap-bg");
    const originalFg = document.getElementById("original-fg");
    const sliderContainer = document.getElementById("comparison-slider");
    const sliderHandle = document.getElementById("slider-handle");
    const imgOverlay = document.getElementById("img-overlay");

    // Model info badge
    const modelInfoBadge = document.getElementById("model-info-badge");
    const modelInfoText = document.getElementById("model-info-text");
    const modelInfoRefresh = document.getElementById("model-info-refresh");
    const modelInfoCopy = document.getElementById("model-info-copy");
    let lastCkptPath = null;
    
    let currentFile = null;
    let originalImageSrc = null;

    // --- Slider Logic ---
    let isDragging = false;

    function initSlider() {
        if (!sliderHandle || !sliderContainer) return;
        
        // Reset position
        updateSliderPosition(sliderContainer.offsetWidth / 2);

        sliderHandle.addEventListener('mousedown', startDrag);
        sliderHandle.addEventListener('touchstart', startDrag);
        
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('touchend', stopDrag);
        
        document.addEventListener('mousemove', drag);
        document.addEventListener('touchmove', drag);
    }

    function startDrag(e) {
        isDragging = true;
        sliderContainer.classList.add('active');
    }

    function stopDrag(e) {
        isDragging = false;
        sliderContainer.classList.remove('active');
    }

    function drag(e) {
        if (!isDragging) return;
        e.preventDefault(); // Prevent scrolling on touch
        
        let clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
        let containerRect = sliderContainer.getBoundingClientRect();
        let x = clientX - containerRect.left;
        
        updateSliderPosition(x);
    }

    function updateSliderPosition(x) {
        let containerWidth = sliderContainer.offsetWidth;
        
        // Clamp
        if (x < 0) x = 0;
        if (x > containerWidth) x = containerWidth;
        
        // Update elements
        sliderHandle.style.left = x + 'px';
        imgOverlay.style.width = x + 'px';
        
        // CRITICAL: Keep inner image width equal to container width to prevent squashing
        originalFg.style.width = containerWidth + 'px';
    }
    
    // Resize observer to handle window resizing
    new ResizeObserver(() => {
        updateSliderPosition(parseFloat(sliderHandle.style.left) || sliderContainer.offsetWidth / 2);
    }).observe(sliderContainer);

    initSlider();
    // --------------------

    if (debugToggle && debugPre) {
        debugToggle.addEventListener('click', () => {
            const hidden = debugPre.style.display === 'none';
            debugPre.style.display = hidden ? 'block' : 'none';
            debugToggle.textContent = hidden ? '隐藏' : '显示';
        });
    }

    async function refreshModelInfo() {
        if (!modelInfoBadge || !modelInfoText) return;

        try {
            modelInfoText.textContent = "模型：读取中…";
            modelInfoBadge.classList.remove('ok', 'warn');

            const response = await fetch(`${API_BASE}/config`, { method: 'GET' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const cfg = await response.json();

            const modelVersion = cfg.model_version ?? "?";
            const ckptPath = cfg.ckpt_path ?? null;
            lastCkptPath = ckptPath;

            if (ckptPath) {
                const shortPath = ckptPath.length > 72 ? ("…" + ckptPath.slice(-72)) : ckptPath;
                modelInfoText.textContent = `模型：${modelVersion} | ${shortPath}`;
                modelInfoBadge.title = `ckpt_path: ${ckptPath}`;
                modelInfoBadge.classList.add('ok');
            } else {
                const resolvedOutDir = cfg.resolved_out_dir ?? "(unknown OUT_DIR)";
                modelInfoText.textContent = `模型：${modelVersion} | 未加载权重`;
                modelInfoBadge.title = `ckpt_path: null\nresolved_out_dir: ${resolvedOutDir}`;
                modelInfoBadge.classList.add('warn');
            }
        } catch (err) {
            lastCkptPath = null;
            modelInfoText.textContent = "模型：无法连接后端";
            modelInfoBadge.title = `无法请求 ${API_BASE}/config：${err}`;
            modelInfoBadge.classList.add('warn');
        }
    }

    if (modelInfoRefresh) {
        modelInfoRefresh.addEventListener('click', (e) => {
            e.preventDefault();
            refreshModelInfo();
        });
    }

    if (modelInfoCopy) {
        modelInfoCopy.addEventListener('click', async (e) => {
            e.preventDefault();
            if (!lastCkptPath) {
                alert('当前没有 ckpt_path 可复制（可能未加载到权重）。');
                return;
            }
            try {
                await navigator.clipboard.writeText(lastCkptPath);
            } catch {
                // Fallback: prompt
                window.prompt('复制 ckpt_path：', lastCkptPath);
            }
        });
    }

    // Load once on startup
    refreshModelInfo();

    uploadArea.addEventListener("click", () => fileInput.click());

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    removeBtn.addEventListener("click", resetUI);

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("请上传图片文件。");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImageSrc = e.target.result; // Store for comparison
            imagePreview.src = originalImageSrc;
            originalFg.src = originalImageSrc; // Set for comparison slider
            
            // Explicitly reset dimensions to avoid stale size
            originalFg.style.width = '';
            originalFg.style.height = '';
            
            uploadArea.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            resultContainer.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = "";
        uploadArea.classList.remove("hidden");
        previewContainer.classList.add("hidden");
        resultContainer.classList.add("hidden");
    }

    predictBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        resultContainer.classList.remove("hidden");
        loader.classList.remove("hidden");
        resultContent.classList.add("hidden");

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Network error");
            const data = await response.json();

            if (data.error) {
                alert("错误: " + data.error);
                resultContainer.classList.add("hidden");
                return;
            }

            displayResult(data);

        } catch (error) {
            alert("分析失败。请检查服务器是否运行在 http://127.0.0.1:5000");
            resultContainer.classList.add("hidden");
            console.error(error);
        } finally {
            loader.classList.add("hidden");
        }
    });

    function displayResult(data) {
        resultContent.classList.remove("hidden");
        
        const isFake = data.class_name === "AI Generated";
        const probFakeVal = data.probabilities["AI Generated"];
        const probRealVal = data.probabilities["Real Photo"];

        if (debugPre) {
            debugPre.textContent = JSON.stringify(data, null, 2);
        }
        
        probReal.textContent = (probRealVal * 100).toFixed(2) + "%";
        probFake.textContent = (probFakeVal * 100).toFixed(2) + "%";
        
        // Heatmap Handling
        if (data.heatmap) {
            heatmapContainer.classList.remove("hidden");
            heatmapBg.src = "data:image/jpeg;base64," + data.heatmap;
            // Ensure original image is set
            if (originalImageSrc) originalFg.src = originalImageSrc;
            
            // Adjust container dimensions to perfectly fit the image aspect ratio
            heatmapBg.onload = () => {
                const natW = heatmapBg.naturalWidth;
                const natH = heatmapBg.naturalHeight;
                if (!natW || !natH) return; // Guard

                // Use the preview container width as reference since it's the parent width constraint
                const containerW = previewContainer.offsetWidth;
                // Or use a max-width if preview container isn't reliable
                const maxW = Math.min(containerW, 800); 
                
                // Calculate displayed height based on aspect ratio
                const displayH = (natH / natW) * maxW;
                
                // Apply dimensions to container
                sliderContainer.style.width = maxW + 'px';
                sliderContainer.style.height = displayH + 'px';
                
                // Center the slider container
                sliderContainer.style.marginLeft = 'auto';
                sliderContainer.style.marginRight = 'auto';
                
                // Ensure the inner original image matches container width exactly
                originalFg.style.width = maxW + 'px';
                originalFg.style.height = displayH + 'px'; // Explicitly set height to avoid distortion
                
                // IMPORTANT: Reset slider pos
                updateSliderPosition(maxW / 2);
            };

            // Reset slider to middle (fallback)
            setTimeout(() => {
                // Trigger resize logic if image already cached
                if (heatmapBg.complete) heatmapBg.onload();
            }, 50);
        } else {
            heatmapContainer.classList.add("hidden");
        }
        
        statusBadge.className = "status-badge " + (isFake ? "fake" : "real");
        statusIcon.className = isFake ? "fa-solid fa-robot" : "fa-solid fa-user-check";
        resultText.textContent = isFake ? "AI 生成" : "真实照片";
        
        meterFill.style.width = `${probFakeVal * 100}%`;
        
        if (probFakeVal > 0.5) {
             meterFill.style.background = "linear-gradient(90deg, #6366f1, #ef4444)";
             probabilityText.textContent = `疑似 AI (${(probFakeVal * 100).toFixed(0)}%)`;
             probabilityText.style.color = "#ef4444";
        } else {
             meterFill.style.background = "linear-gradient(90deg, #10b981, #6366f1)";
             probabilityText.textContent = `疑似 真实 (${(probRealVal * 100).toFixed(0)}%)`;
             probabilityText.style.color = "#10b981";
        }
    }
});
