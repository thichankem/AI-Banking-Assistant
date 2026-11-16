/**
 * AI Banking Assistant - Modern Client v4.5 (Vietcombank Specialized Agentic RAG)
 * Tuân thủ 100%: 0% Regex Law & Zero Ternary Law.
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const clearChatBtn = document.getElementById("clearChatBtn");

    // Stepper DOM Elements
    const nodeStepper = document.getElementById("nodeStepper");
    const stepperDesc = document.getElementById("stepperDesc");
    const agentActions = document.getElementById("agentActions");

    function newSessionId() {
        return Array.from(window.crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, "0")).join("");
    }

    let currentSessionId = newSessionId();

    const initialWelcomeHtml = chatMessages.innerHTML;

    // Thoát ký tự HTML an toàn (0% Regex)
    function escapeHtml(str) {
        if (!str) {
            return "";
        }
        return str
            .split("&").join("&amp;")
            .split("<").join("&lt;")
            .split(">").join("&gt;")
            .split('"').join("&quot;")
            .split("'").join("&#039;");
    }

    // Xử lý in đậm và code inline (0% Regex, Zero Ternary)
    function renderInlineFormatting(str) {
        let safe = escapeHtml(str);

        let boldParts = safe.split("**");
        if (boldParts.length > 1) {
            let boldOut = [];
            for (let i = 0; i < boldParts.length; i++) {
                if (i % 2 === 1) {
                    boldOut.push("<strong>" + boldParts[i] + "</strong>");
                } else {
                    boldOut.push(boldParts[i]);
                }
            }
            safe = boldOut.join("");
        }

        let codeParts = safe.split("`");
        if (codeParts.length > 1) {
            let codeOut = [];
            for (let i = 0; i < codeParts.length; i++) {
                if (i % 2 === 1) {
                    codeOut.push("<code>" + codeParts[i] + "</code>");
                } else {
                    codeOut.push(codeParts[i]);
                }
            }
            safe = codeOut.join("");
        }

        return safe;
    }

    // Parse Markdown (0% Regex, Zero Ternary)
    function parseMarkdown(text) {
        if (!text) {
            return "";
        }
        const lines = text.split("\n");
        let htmlParts = [];
        let inList = false;
        let inTable = false;
        let tableRows = [];

        function flushTable() {
            if (!inTable) {
                return;
            }
            if (tableRows.length === 0) {
                inTable = false;
                tableRows = [];
                return;
            }
            let tblHtml = ['<div class="table-wrap"><table>'];
            let startIdx = 0;

            if (tableRows.length >= 2) {
                tblHtml.push('<thead><tr>');
                for (let c of tableRows[0]) {
                    tblHtml.push("<th>" + renderInlineFormatting(c) + "</th>");
                }
                tblHtml.push('</tr></thead>');
                startIdx = 2;
            }

            tblHtml.push('<tbody>');
            for (let r = startIdx; r < tableRows.length; r++) {
                tblHtml.push('<tr>');
                for (let c of tableRows[r]) {
                    tblHtml.push("<td>" + renderInlineFormatting(c) + "</td>");
                }
                tblHtml.push('</tr>');
            }
            tblHtml.push('</tbody></table></div>');
            htmlParts.push(tblHtml.join(""));
            inTable = false;
            tableRows = [];
        }

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            let trimmed = line.trim();

            if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                let rawCells = trimmed.substring(1, trimmed.length - 1).split("|");
                let cells = [];
                for (let cell of rawCells) {
                    cells.push(cell.trim());
                }
                inTable = true;
                tableRows.push(cells);
                continue;
            } else {
                if (inTable) {
                    flushTable();
                }
            }

            if (trimmed.startsWith("### ")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                htmlParts.push("<h4>" + renderInlineFormatting(trimmed.substring(4)) + "</h4>");
                continue;
            }
            if (trimmed.startsWith("## ")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                htmlParts.push("<h3>" + renderInlineFormatting(trimmed.substring(3)) + "</h3>");
                continue;
            }

            if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                if (!inList) {
                    htmlParts.push("<ul>");
                    inList = true;
                }
                htmlParts.push("<li>" + renderInlineFormatting(trimmed.substring(2)) + "</li>");
                continue;
            } else {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
            }

            if (trimmed === "") {
                continue;
            }

            htmlParts.push("<p>" + renderInlineFormatting(trimmed) + "</p>");
        }

        if (inList) {
            htmlParts.push("</ul>");
        }
        if (inTable) {
            flushTable();
        }

        return htmlParts.join("");
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function resetStepper() {
        agentActions.replaceChildren();
        stepperDesc.textContent = "Đang nhận yêu cầu...";
    }

    function updateStepperProgress(description, action) {
        stepperDesc.textContent = description;
        if (action) {
            const item = document.createElement("li");
            item.textContent = description;
            agentActions.appendChild(item);
        }
    }

    function finishProgress(message = "Đã hoàn tất phản hồi") {
        stepperDesc.textContent = message;
    }

    function appendMessage(sender, content, meta = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + sender + "-message";

        if (sender === "user") {
            msgDiv.innerHTML = `
                <div class="bubble">
                    <p>${escapeHtml(content)}</p>
                </div>
            `;
        } else {
            let sourcesHtml = "";
            if (meta) {
                const combined = new Set();
                for (const citation of meta.citations || []) {
                    if (typeof citation === "string") {
                        combined.add(citation);
                    } else if (citation && citation.source) {
                        combined.add(citation.label + " " + citation.source);
                    }
                }
                if (combined.size === 0) {
                    for (const source of meta.sources || []) {
                        combined.add(source);
                    }
                }

                if (combined.size > 0) {
                    let pills = [];
                    for (let item of Array.from(combined).slice(0, 6)) {
                        pills.push(`<span class="source-tag">${escapeHtml(item)}</span>`);
                    }
                    sourcesHtml = `
                        <div class="sources-card">
                            <div class="sources-header">Tài liệu đã trích dẫn</div>
                            <div class="sources-pills">${pills.join("")}</div>
                        </div>
                    `;
                }
            }

            msgDiv.innerHTML = `
                <div class="bubble">
                    ${parseMarkdown(content)}
                    ${sourcesHtml}
                </div>
            `;
        }

        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    async function submitQueryFallback(cleanQuery) {
        try {
            const res = await fetch("/api/agent/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleanQuery, session_id: currentSessionId })
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: "Lỗi kết nối máy chủ dịch vụ" }));
                let detailMsg = "Không thể xử lý yêu cầu lúc này.";
                if (errData && errData.detail) {
                    detailMsg = errData.detail;
                }
                appendMessage("bot", "**Rất tiếc:** " + detailMsg);
            } else {
                const data = await res.json();
                finishProgress();
                let ans = "Không nhận được phản hồi từ hệ thống.";
                if (data.answer) {
                    ans = data.answer;
                }
                appendMessage("bot", ans, {
                    citations: data.citations,
                    sources: data.sources
                });
            }
        } catch (err) {
            appendMessage("bot", "**Lỗi kết nối:** Không thể liên lạc với máy chủ (" + err.message + ").");
        }
    }

    async function submitQuery(queryText) {
        let cleanQuery = "";
        if (queryText) {
            cleanQuery = queryText.trim();
        }
        if (cleanQuery === "") {
            return;
        }

        appendMessage("user", cleanQuery);
        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        resetStepper();
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }

        // Tạo bubble tạm thời để stream tokens vào
        const botMsgDiv = document.createElement("div");
        botMsgDiv.className = "message bot-message";
        botMsgDiv.innerHTML = `
            <div class="bubble" id="streamingBubble">
                <span id="streamingText" style="color: #94a3b8; font-style: italic;">Đang kết nối agent...</span>
            </div>
        `;
        chatMessages.appendChild(botMsgDiv);
        scrollToBottom();

        const streamingBubble = botMsgDiv.querySelector("#streamingBubble");
        let accumulatedText = "";
        let finalDoneMeta = null;
        let streamStarted = false;

        try {
            const response = await fetch("/api/agent/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: cleanQuery, session_id: currentSessionId })
            });

            if (!response.ok || !response.body) {
                botMsgDiv.remove();
                await submitQueryFallback(cleanQuery);
                return;
            }

            streamStarted = true;
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer = buffer + decoder.decode(value, { stream: true });
                const parts = buffer.split("\n\n");
                buffer = parts.pop();

                for (let part of parts) {
                    const trimmedPart = part.trim();
                    if (trimmedPart.startsWith("data: ")) {
                        const jsonStr = trimmedPart.substring(6).trim();
                        try {
                            const eventData = JSON.parse(jsonStr);
                            const eventType = eventData.event_type;

                            if (eventType === "node_progress") {
                                updateStepperProgress(eventData.description, eventData.action);
                            } else if (eventType === "token") {
                                accumulatedText = accumulatedText + eventData.content;
                                if (streamingBubble) {
                                    streamingBubble.innerHTML = parseMarkdown(accumulatedText);
                                    scrollToBottom();
                                }
                            } else if (eventType === "done") {
                                finalDoneMeta = eventData;
                                finishProgress();
                            }
                        } catch (e) {
                            console.warn("Parse SSE event error:", e);
                        }
                    }
                }
            }

            if (finalDoneMeta) {
                let finalText = accumulatedText;
                if (finalDoneMeta.full_answer) {
                    finalText = finalDoneMeta.full_answer;
                }

                botMsgDiv.remove();
                appendMessage("bot", finalText, {
                    citations: finalDoneMeta.citations,
                    sources: finalDoneMeta.sources
                });
            } else {
                botMsgDiv.remove();
                finishProgress("Kết nối bị gián đoạn");
                appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
            }

        } catch (err) {
            botMsgDiv.remove();
            if (streamStarted) {
                finishProgress("Kết nối bị gián đoạn");
                appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
            } else {
                await submitQueryFallback(cleanQuery);
            }
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery(userInput.value);
    });

    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => {
            currentSessionId = newSessionId();
            chatMessages.innerHTML = initialWelcomeHtml;
            if (nodeStepper) {
                nodeStepper.style.display = "none";
            }
            userInput.value = "";
            userInput.focus();
        });
    }

    if (userInput) {
        userInput.focus();
    }
});
