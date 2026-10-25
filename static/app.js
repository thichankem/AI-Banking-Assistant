/**
 * AI Banking Assistant - Modern Client
 * Tối giản, thanh lịch, an toàn và chuyên nghiệp cho nghiệp vụ Ngân hàng
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const modelStatusPill = document.getElementById("modelStatusPill");
    const chunksCountText = document.getElementById("chunksCountText");
    const suggestionGrid = document.getElementById("suggestionGrid");
    const clearChatBtn = document.getElementById("clearChatBtn");

    // Lưu template chào ban đầu để khôi phục khi bấm "Làm mới"
    const initialWelcomeHtml = chatMessages.innerHTML;

    // 1. Tải thông số cấu hình hệ thống
    async function fetchSystemConfig() {
        try {
            const res = await fetch("/api/config");
            if (!res.ok) return;
            const data = await res.json();
            
            if (modelStatusPill && data.default_model) {
                const textElem = modelStatusPill.querySelector(".status-text");
                if (textElem) textElem.textContent = data.default_model;
            }
            if (chunksCountText && data.points_count !== undefined) {
                chunksCountText.textContent = `${data.points_count.toLocaleString("vi-VN")} văn bản quy chuẩn`;
            }
        } catch (err) {
            console.warn("Không thể tải cấu hình máy chủ:", err);
        }
    }
    fetchSystemConfig();

    // 2. Thoát ký tự HTML an toàn (0% REGEX)
    function escapeHtml(str) {
        if (!str) return "";
        return str
            .split("&").join("&amp;")
            .split("<").join("&lt;")
            .split(">").join("&gt;")
            .split('"').join("&quot;")
            .split("'").join("&#039;");
    }

    // 3. Xử lý định dạng chữ in đậm và code inline (0% REGEX)
    function renderInlineFormatting(str) {
        let safe = escapeHtml(str);

        // Xử lý **in đậm**
        let boldParts = safe.split("**");
        if (boldParts.length > 1) {
            let boldOut = [];
            for (let i = 0; i < boldParts.length; i++) {
                if (i % 2 === 1) {
                    boldOut.push(`<strong>${boldParts[i]}</strong>`);
                } else {
                    boldOut.push(boldParts[i]);
                }
            }
            safe = boldOut.join("");
        }

        // Xử lý `code inline`
        let codeParts = safe.split("`");
        if (codeParts.length > 1) {
            let codeOut = [];
            for (let i = 0; i < codeParts.length; i++) {
                if (i % 2 === 1) {
                    codeOut.push(`<code>${codeParts[i]}</code>`);
                } else {
                    codeOut.push(codeParts[i]);
                }
            }
            safe = codeOut.join("");
        }

        return safe;
    }

    // 4. Định dạng Markdown đơn giản & an toàn cho câu trả lời ngân hàng
    function formatMarkdown(text) {
        if (!text) return "";
        const lines = text.split("\n");
        let htmlParts = [];
        let inList = false;

        for (let i = 0; i < lines.length; i++) {
            let rawLine = lines[i];
            let trimmed = rawLine.trim();

            if (trimmed.startsWith("### ")) {
                if (inList) { htmlParts.push("</ul>"); inList = false; }
                htmlParts.push(`<h4>${renderInlineFormatting(trimmed.substring(4))}</h4>`);
                continue;
            }
            if (trimmed.startsWith("## ")) {
                if (inList) { htmlParts.push("</ul>"); inList = false; }
                htmlParts.push(`<h3>${renderInlineFormatting(trimmed.substring(3))}</h3>`);
                continue;
            }

            // Gạch đầu dòng
            if (trimmed.startsWith("* ") || trimmed.startsWith("- ")) {
                if (!inList) {
                    htmlParts.push("<ul>");
                    inList = true;
                }
                htmlParts.push(`<li>${renderInlineFormatting(trimmed.substring(2))}</li>`);
                continue;
            } else {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
            }

            // Dòng trống
            if (trimmed === "") {
                htmlParts.push("<br>");
                continue;
            }

            // Đoạn văn thường
            htmlParts.push(`<p>${renderInlineFormatting(rawLine)}</p>`);
        }

        if (inList) {
            htmlParts.push("</ul>");
        }

        return htmlParts.join("");
    }

    // 5. Thêm tin nhắn vào giao diện
    function appendMessage(sender, content, meta = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message`;

        const now = new Date();
        const timeStr = now.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });

        if (sender === "user") {
            msgDiv.innerHTML = `
                <div class="message-content">
                    <div class="bubble">
                        <p>${escapeHtml(content)}</p>
                    </div>
                </div>
            `;
        } else {
            let sourcesHtml = "";

            if (meta) {
                const sourcesList = meta.sources || [];
                const citationsList = meta.citations || [];
                const combined = [];

                citationsList.forEach(c => {
                    if (c && !combined.includes(c)) combined.push(c);
                });
                sourcesList.forEach(s => {
                    if (s && !combined.includes(s)) combined.push(s);
                });

                if (combined.length > 0) {
                    const tagsHtml = combined.slice(0, 4).map(src => `<span class="source-tag">${escapeHtml(src)}</span>`).join("");
                    sourcesHtml = `
                        <div class="sources-card">
                            <div class="sources-header">
                                <span>Căn cứ quy chuẩn & Nguồn tham chiếu:</span>
                            </div>
                            <div class="sources-pills">${tagsHtml}</div>
                        </div>
                    `;
                }
            }

            msgDiv.innerHTML = `
                <div class="message-content">
                    <div class="bubble">
                        ${formatMarkdown(content)}
                        ${sourcesHtml}
                    </div>
                </div>
            `;
        }

        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    // Cuộn mượt xuống đáy khung chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 6. Gửi câu hỏi đến Backend
    async function submitQuery(queryText) {
        const cleanQuery = queryText ? queryText.trim() : "";
        if (!cleanQuery) return;

        appendMessage("user", cleanQuery);

        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Hiển thị bong bóng chờ phản hồi
        const typingDiv = document.createElement("div");
        typingDiv.className = "message bot-message";
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="bubble typing-bubble">
                    <div class="typing-dots">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                    <span class="typing-text">Trợ lý AI đang tra cứu dữ liệu quy chuẩn...</span>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleanQuery })
            });

            typingDiv.remove();

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Lỗi kết nối máy chủ dịch vụ" }));
                appendMessage("bot", `**Rất tiếc:** ${err.detail || "Không thể xử lý yêu cầu lúc này."}`);
            } else {
                const data = await res.json();
                appendMessage("bot", data.reply, {
                    latency_ms: data.latency_ms,
                    citations: data.citations,
                    sources: data.sources
                });
            }
        } catch (err) {
            typingDiv.remove();
            appendMessage("bot", `**Lỗi kết nối:** Không thể liên lạc với máy chủ. Vui lòng kiểm tra lại dịch vụ (${err.message}).`);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    // Bắt sự kiện submit form
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery(userInput.value);
    });

    // Bắt sự kiện click gợi ý câu hỏi
    function bindSuggestionChips() {
        const grid = document.getElementById("suggestionGrid");
        if (grid) {
            grid.addEventListener("click", (e) => {
                const chip = e.target.closest(".suggestion-chip");
                if (chip) {
                    const q = chip.getAttribute("data-q");
                    submitQuery(q);
                }
            });
        }
    }
    bindSuggestionChips();

    // Nút làm mới / xóa cuộc hội thoại
    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => {
            chatMessages.innerHTML = initialWelcomeHtml;
            bindSuggestionChips();
            userInput.value = "";
            userInput.focus();
        });
    }

    // Focus vào ô nhập khi vừa mở trang
    if (userInput) {
        userInput.focus();
    }
});
