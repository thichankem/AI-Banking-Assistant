document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const modeSelect = document.getElementById("modeSelect");
    const modelBadge = document.getElementById("modelBadge");

    // Modal elements
    const benchmarkModal = document.getElementById("benchmarkModal");
    const openBenchmarkBtn = document.getElementById("openBenchmarkBtn");
    const closeBenchmarkBtn = document.getElementById("closeBenchmarkBtn");
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Benchmark Run elements
    const runBenchmarkBtn = document.getElementById("runBenchmarkBtn");
    const benchmarkModeSelect = document.getElementById("benchmarkModeSelect");
    const benchmarkSampleSelect = document.getElementById("benchmarkSampleSelect");
    const benchmarkLoading = document.getElementById("benchmarkLoading");
    const benchmarkResultsArea = document.getElementById("benchmarkResultsArea");
    const summaryTableBody = document.getElementById("summaryTableBody");
    const detailedTestCasesList = document.getElementById("detailedTestCasesList");
    const datasetListContainer = document.getElementById("datasetListContainer");

    // Live Eval Modal elements
    const liveEvalModal = document.getElementById("liveEvalModal");
    const closeLiveEvalBtn = document.getElementById("closeLiveEvalBtn");
    const liveEvalContent = document.getElementById("liveEvalContent");

    // Biến lưu hội thoại hiện tại
    let lastUserQuery = "";

    // 1. Lấy thông tin cấu hình từ API
    fetch("/api/config")
        .then(res => res.json())
        .then(cfg => {
            if (cfg.default_model) {
                modelBadge.textContent = cfg.default_model;
            }
        })
        .catch(() => {});

    // 2. Hàm thêm tin nhắn vào khung chat
    function appendMessage(sender, text, isLoading = false, queryContext = "") {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message ${isLoading ? "loading" : ""}`;

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        
        // Định dạng Markdown cơ bản
        let formatted = text
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(/\n/g, "<br>");
            
        bubble.innerHTML = formatted;
        msgDiv.appendChild(bubble);

        // Thêm nút Live Eval nếu là tin nhắn bot đã phản hồi hoàn tất
        if (sender === "bot" && !isLoading && text && !text.startsWith("❌")) {
            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";
            
            const evalBtn = document.createElement("button");
            evalBtn.type = "button";
            evalBtn.className = "btn-live-eval";
            evalBtn.innerHTML = "🔍 Chấm điểm câu này (Live Eval)";
            evalBtn.onclick = () => runLiveEvaluation(queryContext, text);
            
            metaDiv.appendChild(evalBtn);
            msgDiv.appendChild(metaDiv);
        }

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }

    // 3. Xử lý gửi tin nhắn Chat
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        lastUserQuery = text;
        appendMessage("user", text);
        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        const currentMode = modeSelect.value;
        const loadingMsg = appendMessage("bot", `Đang xử lý ở chế độ [${currentMode.toUpperCase()}]...`, true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    mode: currentMode
                })
            });

            loadingMsg.remove();

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Lỗi máy chủ" }));
                appendMessage("bot", `❌ Lỗi: ${err.detail || "Không nhận được phản hồi."}`);
            } else {
                const data = await res.json();
                appendMessage("bot", data.reply, false, text);
            }
        } catch (err) {
            loadingMsg.remove();
            appendMessage("bot", `❌ Lỗi kết nối: ${err.message}`);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    // 4. Modal Benchmark Mở / Đóng
    openBenchmarkBtn.addEventListener("click", () => {
        benchmarkModal.style.display = "flex";
        loadDatasetExplorer();
    });

    closeBenchmarkBtn.addEventListener("click", () => {
        benchmarkModal.style.display = "none";
    });

    closeLiveEvalBtn.addEventListener("click", () => {
        liveEvalModal.style.display = "none";
    });

    // Chuyển Tab
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            tabPanes.forEach(p => p.style.display = "none");

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
                targetPane.style.display = "block";
            }
        });
    });

    // 5. Nạp danh sách bộ test Dataset
    async function loadDatasetExplorer() {
        if (datasetListContainer.children.length > 0) return;
        try {
            const res = await fetch("/api/benchmark/dataset");
            const data = await res.json();
            datasetListContainer.innerHTML = "";
            
            data.forEach((item, idx) => {
                const div = document.createElement("div");
                div.className = "dataset-item";
                div.innerHTML = `
                    <div class="dataset-q">#${item.id} [${item.category.toUpperCase()}]: ${item.question}</div>
                    <div class="dataset-a"><strong>Đáp án chuẩn:</strong> ${item.ground_truth}</div>
                    <div class="dataset-meta">
                        <span>Căn cứ: ${item.expected_articles.join(", ") || "Không có (Câu hỏi bẫy)"}</span> | 
                        <span>Bẫy ảo giác: ${item.is_unanswerable ? "🚨 Có" : "Không"}</span>
                    </div>
                `;
                datasetListContainer.appendChild(div);
            });
        } catch (e) {
            datasetListContainer.innerHTML = "<p style='color: #f87171;'>Không thể tải bộ dữ liệu kiểm chuẩn.</p>";
        }
    }

    // 6. Xử lý Chạy Benchmark Toàn Diện
    runBenchmarkBtn.addEventListener("click", async () => {
        const mode = benchmarkModeSelect.value;
        const maxCases = parseInt(benchmarkSampleSelect.value);

        runBenchmarkBtn.disabled = true;
        benchmarkLoading.style.display = "flex";
        benchmarkResultsArea.style.display = "none";

        try {
            const res = await fetch("/api/benchmark/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: mode,
                    max_cases: maxCases
                })
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Lỗi chạy benchmark" }));
                alert(`Lỗi: ${err.detail || "Không thể thực thi benchmark."}`);
                return;
            }

            const data = await res.json();
            renderBenchmarkResults(data, mode);
        } catch (err) {
            alert(`Lỗi kết nối khi chạy benchmark: ${err.message}`);
        } finally {
            runBenchmarkBtn.disabled = false;
            benchmarkLoading.style.display = "none";
        }
    });

    // 7. Hiển thị Kết quả Benchmark
    function renderBenchmarkResults(data, mode) {
        benchmarkResultsArea.style.display = "block";

        let bSummary, rSummary, isCompare = (mode === "compare");

        if (isCompare) {
            bSummary = data.baseline_summary;
            rSummary = data.rag_summary;
            const diff = data.improvements;

            // Update KPI cards
            document.getElementById("kpiFaithfulness").textContent = `${(rSummary.mean_faithfulness * 100).toFixed(1)}%`;
            document.getElementById("kpiFaithfulnessDelta").innerHTML = `Tăng <strong class="delta-pos">+${(diff.faithfulness_delta * 100).toFixed(1)}%</strong> so với Baseline`;

            document.getElementById("kpiTriad").textContent = rSummary.mean_rag_triad.toFixed(3);
            document.getElementById("kpiTriadDelta").innerHTML = `Baseline: ${bSummary.mean_rag_triad.toFixed(3)} (<span class="delta-pos">+${diff.rag_triad_delta.toFixed(3)}</span>)`;

            document.getElementById("kpiRagas").textContent = rSummary.ragas_overall_harmonic.toFixed(3);
            document.getElementById("kpiRagasDelta").innerHTML = `Baseline: ${bSummary.ragas_overall_harmonic.toFixed(3)} (<span class="delta-pos">+${diff.ragas_harmonic_delta.toFixed(3)}</span>)`;

            document.getElementById("kpiRejection").textContent = `${(rSummary.negative_rejection_rate * 100).toFixed(1)}%`;
            document.getElementById("kpiRejectionDelta").innerHTML = `Baseline: ${(bSummary.negative_rejection_rate * 100).toFixed(1)}% (<span class="delta-pos">+${(diff.rejection_rate_delta * 100).toFixed(1)}%</span>)`;

            // Render Table
            summaryTableBody.innerHTML = `
                <tr>
                    <td><strong>Faithfulness (Độ trung thực - Chống ảo giác)</strong></td>
                    <td>${(bSummary.mean_faithfulness * 100).toFixed(1)}%</td>
                    <td><strong class="delta-pos">${(rSummary.mean_faithfulness * 100).toFixed(1)}%</strong></td>
                    <td><span class="delta-pos">+${(diff.faithfulness_delta * 100).toFixed(1)}%</span></td>
                </tr>
                <tr>
                    <td><strong>Answer Correctness (Độ chuẩn xác số liệu)</strong></td>
                    <td>${(bSummary.mean_correctness * 100).toFixed(1)}%</td>
                    <td><strong class="delta-pos">${(rSummary.mean_correctness * 100).toFixed(1)}%</strong></td>
                    <td><span class="delta-pos">+${(diff.correctness_delta * 100).toFixed(1)}%</span></td>
                </tr>
                <tr>
                    <td><strong>Negative Rejection Rate (Bắt bẫy câu hỏi ngoài phạm vi)</strong></td>
                    <td>${(bSummary.negative_rejection_rate * 100).toFixed(1)}%</td>
                    <td><strong class="delta-pos">${(rSummary.negative_rejection_rate * 100).toFixed(1)}%</strong></td>
                    <td><span class="delta-pos">+${(diff.rejection_rate_delta * 100).toFixed(1)}%</span></td>
                </tr>
                <tr>
                    <td><strong>Citation Accuracy (Trích dẫn đúng Điều, Khoản)</strong></td>
                    <td>${(bSummary.mean_citation_accuracy * 100).toFixed(1)}%</td>
                    <td><strong class="delta-pos">${(rSummary.mean_citation_accuracy * 100).toFixed(1)}%</strong></td>
                    <td><span class="delta-pos">+${(diff.citations_delta * 100).toFixed(1)}%</span></td>
                </tr>
                <tr>
                    <td><strong>TruLens RAG Triad Score (Context + Faithfulness + Relevance)</strong></td>
                    <td>${bSummary.mean_rag_triad.toFixed(4)}</td>
                    <td><strong class="delta-pos">${rSummary.mean_rag_triad.toFixed(4)}</strong></td>
                    <td><span class="delta-pos">+${diff.rag_triad_delta.toFixed(4)}</span></td>
                </tr>
                <tr>
                    <td><strong>RAGAS Harmonic Mean (Điểm điều hòa toàn diện)</strong></td>
                    <td>${bSummary.ragas_overall_harmonic.toFixed(4)}</td>
                    <td><strong class="delta-pos">${rSummary.ragas_overall_harmonic.toFixed(4)}</strong></td>
                    <td><span class="delta-pos">+${diff.ragas_harmonic_delta.toFixed(4)}</span></td>
                </tr>
                <tr>
                    <td><strong>Độ trễ trung bình (Average Latency)</strong></td>
                    <td>${bSummary.average_latency_ms.toFixed(0)} ms</td>
                    <td>${rSummary.average_latency_ms.toFixed(0)} ms</td>
                    <td>+${(rSummary.average_latency_ms - bSummary.average_latency_ms).toFixed(0)} ms</td>
                </tr>
            `;

            renderTestCasesAccordion(rSummary.detailed_results);
        } else {
            // Chế độ đơn lẻ
            rSummary = data;
            document.getElementById("kpiFaithfulness").textContent = `${(rSummary.mean_faithfulness * 100).toFixed(1)}%`;
            document.getElementById("kpiFaithfulnessDelta").textContent = `Chế độ: ${rSummary.mode}`;
            document.getElementById("kpiTriad").textContent = rSummary.mean_rag_triad.toFixed(3);
            document.getElementById("kpiTriadDelta").textContent = "Trung bình 3 cạnh Tam giác RAG";
            document.getElementById("kpiRagas").textContent = rSummary.ragas_overall_harmonic.toFixed(3);
            document.getElementById("kpiRagasDelta").textContent = "Chuẩn Harmonic Mean Ragas";
            document.getElementById("kpiRejection").textContent = `${(rSummary.negative_rejection_rate * 100).toFixed(1)}%`;
            document.getElementById("kpiRejectionDelta").textContent = "Bẫy câu hỏi chưa quy định";

            summaryTableBody.innerHTML = `
                <tr>
                    <td><strong>Faithfulness</strong></td>
                    <td colspan="3"><strong class="delta-pos">${(rSummary.mean_faithfulness * 100).toFixed(1)}%</strong></td>
                </tr>
                <tr>
                    <td><strong>Correctness</strong></td>
                    <td colspan="3"><strong class="delta-pos">${(rSummary.mean_correctness * 100).toFixed(1)}%</strong></td>
                </tr>
                <tr>
                    <td><strong>RAG Triad Score</strong></td>
                    <td colspan="3">${rSummary.mean_rag_triad.toFixed(4)}</td>
                </tr>
                <tr>
                    <td><strong>RAGAS Harmonic Mean</strong></td>
                    <td colspan="3"><strong class="delta-pos">${rSummary.ragas_overall_harmonic.toFixed(4)}</strong></td>
                </tr>
                <tr>
                    <td><strong>Độ trễ trung bình</strong></td>
                    <td colspan="3">${rSummary.average_latency_ms.toFixed(0)} ms (P95: ${rSummary.p95_latency_ms.toFixed(0)} ms)</td>
                </tr>
            `;

            renderTestCasesAccordion(rSummary.detailed_results);
        }
    }

    // 8. Render Accordion danh sách câu hỏi test
    function renderTestCasesAccordion(cases) {
        detailedTestCasesList.innerHTML = "";
        cases.forEach(c => {
            const item = document.createElement("div");
            item.className = "test-case-item";

            const isPass = c.faithfulness >= 0.8;
            const badgeClass = isPass ? "score-pass" : "score-warn";

            item.innerHTML = `
                <div class="test-case-header">
                    <span class="tc-title"><strong>[${c.test_id}]</strong> ${c.question}</span>
                    <div class="tc-badges">
                        <span class="score-badge ${badgeClass}">Faith: ${(c.faithfulness * 100).toFixed(0)}%</span>
                        <span class="score-badge">Triad: ${c.rag_triad_score.toFixed(2)}</span>
                    </div>
                </div>
                <div class="test-case-body">
                    <div><strong>Đáp án chuẩn:</strong> ${c.ground_truth}</div>
                    <div><strong>Câu trả lời của AI:</strong> <br><em>${c.actual_reply}</em></div>
                    <div><strong>Trích dẫn phát hiện:</strong> <code>${c.found_citations.join(", ") || "Không có"}</code></div>
                    <div style="color: #38bdf8;"><strong>Nhận xét Giám khảo:</strong> ${c.reasoning}</div>
                </div>
            `;
            detailedTestCasesList.appendChild(item);
        });
    }

    // 9. Thực thi Live Evaluation cho 1 tin nhắn
    async function runLiveEvaluation(query, reply) {
        liveEvalModal.style.display = "flex";
        liveEvalContent.innerHTML = `
            <div class="benchmark-loading">
                <div class="spinner"></div>
                <p>Đang thẩm định câu trả lời bằng LLM Judge...</p>
            </div>
        `;

        try {
            const res = await fetch("/api/benchmark/evaluate-single", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: query,
                    actual_reply: reply
                })
            });

            if (!res.ok) throw new Error("Lỗi gọi API thẩm định");
            const data = await res.json();

            liveEvalContent.innerHTML = `
                <div style="background: #0f172a; padding: 12px; border-radius: 8px;">
                    <p><strong>Câu hỏi:</strong> ${data.question}</p>
                </div>
                <div class="kpi-grid" style="margin: 10px 0;">
                    <div class="kpi-card">
                        <div class="kpi-title">Faithfulness (Chống ảo giác)</div>
                        <div class="kpi-value highlight">${(data.faithfulness * 100).toFixed(0)}%</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">TruLens RAG Triad</div>
                        <div class="kpi-value">${data.rag_triad_score.toFixed(3)}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">RAGAS Harmonic</div>
                        <div class="kpi-value highlight">${data.ragas_harmonic_score.toFixed(3)}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Trích Dẫn Căn Cứ</div>
                        <div class="kpi-value">${data.found_citations.length > 0 ? "Đạt" : "0"}</div>
                    </div>
                </div>
                <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                    <p style="color: #38bdf8; font-weight: 600; margin-bottom: 6px;">Lời phê của Giám khảo AI:</p>
                    <p style="color: #cbd5e1; font-style: italic;">${data.reasoning}</p>
                </div>
            `;
        } catch (e) {
            liveEvalContent.innerHTML = `<p style="color: #f87171;">Lỗi khi thẩm định: ${e.message}</p>`;
        }
    }
});
