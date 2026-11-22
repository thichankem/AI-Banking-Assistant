/**
 * A-Bank web chat
 * Tuân thủ 100%: 0% Regex Law & Zero Ternary Law.
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatContainer = document.getElementById("chatContainer");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const clearChatBtn = document.getElementById("clearChatBtn");

    // Stepper DOM Elements
    const nodeStepper = document.getElementById("nodeStepper");
    const stepperDesc = document.getElementById("stepperDesc");
    const agentActions = document.getElementById("agentActions");
    const agentPlan = document.getElementById("agentPlan");

    // Attachment & Voice DOM Elements
    const chatComposer = document.getElementById("chatComposer");
    const composerStatusBanner = document.getElementById("composerStatusBanner");
    const attachmentPreviewBar = document.getElementById("attachmentPreviewBar");
    const imageUploadInput = document.getElementById("imageUploadInput");
    const attachImageBtn = document.getElementById("attachImageBtn");
    const voiceInputBtn = document.getElementById("voiceInputBtn");

    function newSessionId() {
        return Array.from(window.crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, "0")).join("");
    }

    const sessionKey = "banking-agent-session";
    let currentSessionId = sessionStorage.getItem(sessionKey) || newSessionId();
    sessionStorage.setItem(sessionKey, currentSessionId);
    let pendingCard = null;
    let busy = false;

    const MAX_ATTACHMENTS = 3;
    const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
    const ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"];
    const ALLOWED_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif"];
    let pendingAttachments = [];
    let statusBannerTimer = null;

    let speechRecognition = null;
    let isRecording = false;
    let voiceBaseText = "";
    let voiceCapturedText = "";

    if (window.ABankDebugInspector) {
        window.ABankDebugInspector.init({ sessionId: currentSessionId });
    }

    function exitWelcomeState() {
        if (chatContainer) {
            chatContainer.classList.remove("is-empty");
        }
    }

    function enterWelcomeState() {
        if (chatContainer) {
            chatContainer.classList.add("is-empty");
        }
    }

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

    function isDigitsOnly(str) {
        if (!str || str.length === 0) {
            return false;
        }
        for (let i = 0; i < str.length; i++) {
            const ch = str.charAt(i);
            if (ch < "0" || ch > "9") {
                return false;
            }
        }
        return true;
    }

    // Chuyển các nhãn [Nguồn X] hoặc [Nguồn 1, 2] thành nút bấm xem chi tiết (0% Regex, Zero Ternary)
    function linkifyCitationMarkers(safeHtml, citationByIndex) {
        if (!safeHtml || !citationByIndex || citationByIndex.size === 0) {
            return safeHtml;
        }
        let out = [];
        let cursor = 0;
        while (cursor < safeHtml.length) {
            const openIdx = safeHtml.indexOf("[Nguồn", cursor);
            if (openIdx < 0) {
                out.push(safeHtml.slice(cursor));
                break;
            }
            const lastTagOpen = safeHtml.lastIndexOf("<", openIdx);
            const lastTagClose = safeHtml.lastIndexOf(">", openIdx);
            if (lastTagOpen > lastTagClose) {
                out.push(safeHtml.slice(cursor, openIdx + 1));
                cursor = openIdx + 1;
                continue;
            }
            const closeIdx = safeHtml.indexOf("]", openIdx);
            if (closeIdx < 0 || closeIdx - openIdx > 80) {
                out.push(safeHtml.slice(cursor, openIdx + 1));
                cursor = openIdx + 1;
                continue;
            }
            const inner = safeHtml.slice(openIdx + 1, closeIdx);
            const normalizedParts = inner.split(";").join(",").split(",");
            const buttons = [];
            let allValid = true;
            for (let p = 0; p < normalizedParts.length; p++) {
                let token = normalizedParts[p].trim();
                if (!token) {
                    continue;
                }
                if (token.startsWith("Nguồn ")) {
                    token = token.slice(6).trim();
                } else if (token.startsWith("Nguồn")) {
                    token = token.slice(5).trim();
                } else if (p === 0) {
                    allValid = false;
                    break;
                }
                if (token.indexOf(" ") >= 0) {
                    token = token.split(" ")[0].trim();
                }
                if (!isDigitsOnly(token)) {
                    allValid = false;
                    break;
                }
                const num = Number(token);
                if (!citationByIndex.has(num)) {
                    allValid = false;
                    break;
                }
                buttons.push(
                    '<button type="button" class="inline-citation-btn" data-cite-idx="'
                    + num
                    + '" title="Xem đoạn gốc Nguồn '
                    + num
                    + '">[Nguồn '
                    + num
                    + ']</button>'
                );
            }
            if (allValid && buttons.length > 0) {
                out.push(safeHtml.slice(cursor, openIdx));
                out.push(buttons.join(" "));
                cursor = closeIdx + 1;
            } else {
                out.push(safeHtml.slice(cursor, openIdx + 1));
                cursor = openIdx + 1;
            }
        }
        return out.join("");
    }

    // Xử lý in đậm, code inline và nút trích dẫn (0% Regex, Zero Ternary)
    function renderInlineFormatting(str, citationByIndex = null) {
        let safe = escapeHtml(str);
        safe = safe
            .split("&lt;br&gt;").join("<br>")
            .split("&lt;br/&gt;").join("<br>")
            .split("&lt;br /&gt;").join("<br>")
            .split("&lt;BR&gt;").join("<br>")
            .split("&lt;BR/&gt;").join("<br>")
            .split("&lt;BR /&gt;").join("<br>");

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

        if (citationByIndex && citationByIndex.size > 0) {
            safe = linkifyCitationMarkers(safe, citationByIndex);
        }

        return safe;
    }

    // Parse Markdown (0% Regex, Zero Ternary)
    function parseMarkdown(text, citationByIndex = null) {
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
                    tblHtml.push("<th>" + renderInlineFormatting(c, citationByIndex) + "</th>");
                }
                tblHtml.push('</tr></thead>');
                startIdx = 2;
            }

            tblHtml.push('<tbody>');
            for (let r = startIdx; r < tableRows.length; r++) {
                tblHtml.push('<tr>');
                for (let c of tableRows[r]) {
                    tblHtml.push("<td>" + renderInlineFormatting(c, citationByIndex) + "</td>");
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
                htmlParts.push("<h4>" + renderInlineFormatting(trimmed.substring(4), citationByIndex) + "</h4>");
                continue;
            }
            if (trimmed.startsWith("## ")) {
                if (inList) {
                    htmlParts.push("</ul>");
                    inList = false;
                }
                htmlParts.push("<h3>" + renderInlineFormatting(trimmed.substring(3), citationByIndex) + "</h3>");
                continue;
            }

            if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                if (!inList) {
                    htmlParts.push("<ul>");
                    inList = true;
                }
                htmlParts.push("<li>" + renderInlineFormatting(trimmed.substring(2), citationByIndex) + "</li>");
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

            htmlParts.push("<p>" + renderInlineFormatting(trimmed, citationByIndex) + "</p>");
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

    function renderPlan(plan, markCompleted = false) {
        if (!Array.isArray(plan)) {
            return;
        }
        const labels = {
            pending: "Chờ xử lý", in_progress: "Đang xử lý",
            done: "Hoàn tất", blocked: "Cần bổ sung", canceled: "Đã hủy"
        };
        const items = plan.map(step => {
            const item = document.createElement("li");
            let statusKey = "pending";
            if (step && step.status) {
                statusKey = String(step.status);
            }
            if (markCompleted && (statusKey === "pending" || statusKey === "in_progress")) {
                statusKey = "done";
            }
            let statusLabel = statusKey;
            if (Object.prototype.hasOwnProperty.call(labels, statusKey)) {
                statusLabel = labels[statusKey];
            }
            let taskText = "";
            if (step && step.task) {
                taskText = String(step.task);
            }
            item.textContent = taskText + " · " + statusLabel;
            return item;
        });
        agentPlan.replaceChildren(...items);
    }

    function resetStepper() {
        renderPlan([], false);
        agentActions.replaceChildren();
        stepperDesc.textContent = "Đang nhận yêu cầu...";
    }

    function updateStepperProgress(description, action) {
        stepperDesc.textContent = description;
        if (action) {
            const lastItem = agentActions.lastElementChild;
            if (lastItem && lastItem.textContent === description) {
                return;
            }
            const item = document.createElement("li");
            item.textContent = description;
            agentActions.appendChild(item);
        }
    }

    function finishProgress(message = "Đã hoàn tất phản hồi") {
        stepperDesc.textContent = message;
    }

    function finishProgressForStatus(auditStatus, turnStatus) {
        if (turnStatus === "awaiting_input" || auditStatus === "AWAITING_INPUT" || auditStatus === "NEEDS_INPUT") {
            finishProgress("Đang chờ lựa chọn của bạn để tiếp tục");
            return;
        }
        if (auditStatus === "CANCELED" || turnStatus === "canceled") {
            finishProgress("Đã hủy yêu cầu theo lựa chọn của bạn");
            return;
        }
        if (auditStatus === "BLOCKED") {
            finishProgress("Yêu cầu đã bị chặn bởi bộ lọc an toàn");
            return;
        }
        if (auditStatus === "PASS") {
            finishProgress("Đã hoàn tất và kiểm duyệt phản hồi");
            return;
        }
        if (auditStatus === "ERROR" || auditStatus === "FAIL") {
            finishProgress("Đã kết thúc xử lý (dữ liệu hiện có bị giới hạn)");
            return;
        }
        finishProgress("Đã hoàn tất phản hồi");
    }

    let lightboxElements = null;
    let lightboxState = { images: [], activeIndex: 0 };

    function formatImageMetaSummary(item) {
        if (!item) {
            return "";
        }
        const subParts = [];
        if (item.document_name) {
            subParts.push(item.document_name);
        }
        if (item.page_number) {
            subParts.push("Trang " + item.page_number);
        }
        if (typeof item.region_index === "number" && item.region_index >= 0 && typeof item.region_count === "number" && item.region_count > 1) {
            subParts.push("Khung " + (item.region_index + 1) + "/" + item.region_count);
        }
        return subParts.join(" · ");
    }

    function ensureLightbox() {
        if (lightboxElements) {
            return lightboxElements;
        }
        const overlay = document.createElement("div");
        overlay.className = "image-lightbox-overlay";
        overlay.hidden = true;

        const dialog = document.createElement("div");
        dialog.className = "image-lightbox-content";

        const topbar = document.createElement("div");
        topbar.className = "image-lightbox-topbar";

        const counterEl = document.createElement("span");
        counterEl.className = "image-lightbox-counter";

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "image-lightbox-close";
        closeBtn.textContent = "×";
        closeBtn.setAttribute("aria-label", "Đóng ảnh phóng to");

        topbar.append(counterEl, closeBtn);

        const stage = document.createElement("div");
        stage.className = "image-lightbox-stage";

        const prevBtn = document.createElement("button");
        prevBtn.type = "button";
        prevBtn.className = "image-lightbox-nav image-lightbox-prev";
        prevBtn.textContent = "‹";
        prevBtn.setAttribute("aria-label", "Ảnh trước");
        prevBtn.title = "Ảnh trước (←)";

        const fullImg = document.createElement("img");
        fullImg.className = "image-lightbox-img";
        fullImg.alt = "";

        const nextBtn = document.createElement("button");
        nextBtn.type = "button";
        nextBtn.className = "image-lightbox-nav image-lightbox-next";
        nextBtn.textContent = "›";
        nextBtn.setAttribute("aria-label", "Ảnh tiếp theo");
        nextBtn.title = "Ảnh tiếp theo (→)";

        stage.append(prevBtn, fullImg, nextBtn);

        const metaBox = document.createElement("div");
        metaBox.className = "image-lightbox-meta";

        dialog.append(topbar, stage, metaBox);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        function renderLightboxImage() {
            const images = lightboxState.images;
            const total = images.length;
            const idx = lightboxState.activeIndex;
            const item = images[idx];
            if (!item) {
                return;
            }

            let imgAlt = "Hình ảnh minh họa";
            if (item.caption) {
                imgAlt = item.caption;
            }
            fullImg.src = item.url;
            fullImg.alt = imgAlt;

            if (total <= 1) {
                counterEl.hidden = true;
                prevBtn.hidden = true;
                nextBtn.hidden = true;
            } else {
                counterEl.hidden = false;
                counterEl.textContent = "Ảnh " + (idx + 1) + " / " + total;
                prevBtn.hidden = false;
                nextBtn.hidden = false;
                prevBtn.disabled = idx <= 0;
                nextBtn.disabled = idx >= total - 1;
            }

            metaBox.replaceChildren();

            const badgesRow = document.createElement("div");
            badgesRow.className = "image-lightbox-badges";
            if (item.citation_label) {
                const citeBadge = document.createElement("span");
                citeBadge.className = "evidence-image-badge";
                citeBadge.textContent = item.citation_label;
                badgesRow.appendChild(citeBadge);
            }
            const typeBadge = document.createElement("span");
            typeBadge.className = "evidence-image-type";
            typeBadge.textContent = regionTypeLabel(item.region_type);
            badgesRow.appendChild(typeBadge);
            metaBox.appendChild(badgesRow);

            const titleEl = document.createElement("div");
            titleEl.className = "image-lightbox-title";
            titleEl.textContent = imgAlt;
            metaBox.appendChild(titleEl);

            const metaSummary = formatImageMetaSummary(item);
            if (metaSummary) {
                const subEl = document.createElement("div");
                subEl.className = "image-lightbox-sub";
                subEl.textContent = metaSummary;
                metaBox.appendChild(subEl);
            }
        }

        function stepLightbox(delta) {
            const nextIdx = lightboxState.activeIndex + delta;
            if (nextIdx >= 0 && nextIdx < lightboxState.images.length) {
                lightboxState.activeIndex = nextIdx;
                renderLightboxImage();
            }
        }

        function closeLightbox() {
            overlay.hidden = true;
            fullImg.src = "";
        }

        prevBtn.addEventListener("click", () => {
            stepLightbox(-1);
        });
        nextBtn.addEventListener("click", () => {
            stepLightbox(1);
        });
        closeBtn.addEventListener("click", closeLightbox);
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) {
                closeLightbox();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (overlay.hidden) {
                return;
            }
            if (event.key === "Escape") {
                closeLightbox();
            } else if (event.key === "ArrowLeft") {
                stepLightbox(-1);
            } else if (event.key === "ArrowRight") {
                stepLightbox(1);
            }
        });

        lightboxElements = { overlay, fullImg, metaBox, renderLightboxImage };
        return lightboxElements;
    }

    function openImageLightbox(imagesList, startIndex = 0) {
        if (!Array.isArray(imagesList) || imagesList.length === 0) {
            return;
        }
        const box = ensureLightbox();
        let idx = startIndex;
        if (typeof idx !== "number" || idx < 0) {
            idx = 0;
        } else if (idx >= imagesList.length) {
            idx = imagesList.length - 1;
        }
        lightboxState = { images: imagesList, activeIndex: idx };
        box.renderLightboxImage();
        box.overlay.hidden = false;
    }

    function regionTypeLabel(regionType) {
        if (regionType === "app_step") {
            return "Thao tác App";
        }
        if (regionType === "slide_visual") {
            return "Hướng dẫn trực quan";
        }
        if (regionType === "table") {
            return "Bảng biểu";
        }
        if (regionType === "visual_card") {
            return "Thẻ chỉ số";
        }
        return "Minh họa";
    }

    function buildEvidenceImagesElement(images) {
        if (!Array.isArray(images) || images.length === 0) {
            return null;
        }
        const validImages = [];
        for (const item of images) {
            if (item && item.url) {
                validImages.push(item);
            }
        }
        if (validImages.length === 0) {
            return null;
        }

        const firstItem = validImages[0];
        const total = validImages.length;

        const card = document.createElement("div");
        card.className = "evidence-images-card";

        const header = document.createElement("div");
        header.className = "evidence-images-header";

        const headerTitle = document.createElement("span");
        headerTitle.textContent = "Hình ảnh minh họa liên quan (" + total + ")";
        header.appendChild(headerTitle);
        card.appendChild(header);

        const singleWrap = document.createElement("div");
        singleWrap.className = "evidence-images-single";

        const tile = document.createElement("button");
        tile.type = "button";
        if (total > 1) {
            tile.className = "evidence-image-item has-more";
        } else {
            tile.className = "evidence-image-item";
        }

        const thumbWrap = document.createElement("div");
        thumbWrap.className = "evidence-image-thumb-wrap";

        const imgEl = document.createElement("img");
        imgEl.src = firstItem.url;
        imgEl.loading = "lazy";
        let imgAlt = "Hình ảnh minh họa";
        if (firstItem.caption) {
            imgAlt = firstItem.caption;
        }
        imgEl.alt = imgAlt;
        thumbWrap.appendChild(imgEl);

        if (total > 1) {
            const moreBadge = document.createElement("span");
            moreBadge.className = "evidence-image-more-badge";
            moreBadge.textContent = "+" + (total - 1) + " ảnh";
            thumbWrap.appendChild(moreBadge);
        }

        const metaDiv = document.createElement("div");
        metaDiv.className = "evidence-image-meta";

        const topRow = document.createElement("div");
        topRow.className = "evidence-image-badges";

        if (firstItem.citation_label) {
            const citeBadge = document.createElement("span");
            citeBadge.className = "evidence-image-badge";
            citeBadge.textContent = firstItem.citation_label;
            topRow.appendChild(citeBadge);
        }

        const typeBadge = document.createElement("span");
        typeBadge.className = "evidence-image-type";
        typeBadge.textContent = regionTypeLabel(firstItem.region_type);
        topRow.appendChild(typeBadge);

        if (total > 1) {
            const countPill = document.createElement("span");
            countPill.className = "evidence-image-count-pill";
            countPill.textContent = "1/" + total;
            topRow.appendChild(countPill);
        }

        const capEl = document.createElement("div");
        capEl.className = "evidence-image-caption";
        const metaSummary = formatImageMetaSummary(firstItem);
        if (metaSummary) {
            capEl.textContent = metaSummary;
        } else {
            capEl.textContent = imgAlt;
        }

        metaDiv.append(topRow, capEl);
        tile.append(thumbWrap, metaDiv);

        tile.addEventListener("click", () => {
            openImageLightbox(validImages, 0);
        });
        singleWrap.appendChild(tile);

        card.appendChild(singleWrap);
        return card;
    }

    let citationDrawerElements = null;
    let citationDrawerState = { citations: [], activePos: 0 };

    function chunkTypeLabel(chunkType) {
        if (chunkType === "table") {
            return "Bảng biểu";
        }
        if (chunkType === "multimodal") {
            return "Đa phương tiện";
        }
        return "Văn bản";
    }

    function buildPdfPageUrl(citation, pageNumber) {
        if (!citation) {
            return "";
        }
        const rawSource = String(citation.source || "").split("\\").join("/").trim();
        let baseUrl = "";
        if (rawSource.toLowerCase().endsWith(".pdf")) {
            baseUrl = "/api/documents/pdf?source=" + encodeURIComponent(rawSource);
        } else if (citation.pdf_url) {
            const rawUrl = String(citation.pdf_url).trim();
            const hashIdx = rawUrl.indexOf("#");
            if (hashIdx >= 0) {
                baseUrl = rawUrl.slice(0, hashIdx);
            } else {
                baseUrl = rawUrl;
            }
        }
        if (!baseUrl) {
            return "";
        }
        let targetPage = 0;
        if (typeof pageNumber === "number" && pageNumber > 0) {
            targetPage = pageNumber;
        } else if (Array.isArray(citation.page_numbers) && citation.page_numbers.length > 0) {
            const first = Number(citation.page_numbers[0]);
            if (Number.isInteger(first) && first > 0) {
                targetPage = first;
            }
        }
        if (targetPage > 0) {
            return baseUrl + "#page=" + targetPage;
        }
        return baseUrl;
    }

    function ensureCitationDrawer() {
        if (citationDrawerElements) {
            return citationDrawerElements;
        }
        const backdrop = document.createElement("div");
        backdrop.className = "citation-drawer-backdrop";
        backdrop.hidden = true;

        const panel = document.createElement("aside");
        panel.className = "citation-drawer";
        panel.setAttribute("role", "dialog");
        panel.setAttribute("aria-label", "Chi tiết tài liệu trích dẫn");

        const header = document.createElement("div");
        header.className = "citation-drawer-header";

        const headerLeft = document.createElement("div");
        headerLeft.className = "citation-drawer-header-left";
        const badgeEl = document.createElement("span");
        badgeEl.className = "citation-drawer-badge";
        const typeEl = document.createElement("span");
        typeEl.className = "citation-drawer-type";
        headerLeft.append(badgeEl, typeEl);

        const headerRight = document.createElement("div");
        headerRight.className = "citation-drawer-nav";
        const prevBtn = document.createElement("button");
        prevBtn.type = "button";
        prevBtn.className = "citation-drawer-nav-btn";
        prevBtn.textContent = "←";
        prevBtn.title = "Nguồn trước";
        const nextBtn = document.createElement("button");
        nextBtn.type = "button";
        nextBtn.className = "citation-drawer-nav-btn";
        nextBtn.textContent = "→";
        nextBtn.title = "Nguồn tiếp theo";
        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "citation-drawer-close";
        closeBtn.textContent = "×";
        closeBtn.setAttribute("aria-label", "Đóng panel trích dẫn");
        headerRight.append(prevBtn, nextBtn, closeBtn);

        header.append(headerLeft, headerRight);

        const tabsRow = document.createElement("div");
        tabsRow.className = "citation-drawer-tabs";

        const contentArea = document.createElement("div");
        contentArea.className = "citation-drawer-scroll";

        const metaBox = document.createElement("div");
        metaBox.className = "citation-drawer-meta";

        const actionsBox = document.createElement("div");
        actionsBox.className = "citation-drawer-actions";
        actionsBox.hidden = true;

        const bodyBox = document.createElement("div");
        bodyBox.className = "citation-drawer-body";

        const bodyHeader = document.createElement("div");
        bodyHeader.className = "citation-drawer-body-header";
        const bodyTitle = document.createElement("span");
        bodyTitle.textContent = "Đoạn văn bản gốc";
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "citation-copy-btn";
        copyBtn.textContent = "Sao chép";
        bodyHeader.append(bodyTitle, copyBtn);

        const excerptEl = document.createElement("div");
        excerptEl.className = "citation-drawer-excerpt";
        bodyBox.append(bodyHeader, excerptEl);

        const imagesBox = document.createElement("div");
        imagesBox.className = "citation-drawer-images";

        contentArea.append(metaBox, actionsBox, bodyBox, imagesBox);
        panel.append(header, tabsRow, contentArea);
        backdrop.appendChild(panel);
        document.body.appendChild(backdrop);

        function closeDrawer() {
            backdrop.hidden = true;
        }

        function renderActiveCitation() {
            const citations = citationDrawerState.citations;
            const pos = citationDrawerState.activePos;
            const item = citations[pos];
            if (!item) {
                return;
            }
            let badgeText = "Nguồn trích dẫn";
            if (item.label) {
                badgeText = item.label;
            } else if (item.index) {
                badgeText = "[Nguồn " + item.index + "]";
            }
            badgeEl.textContent = badgeText;
            typeEl.textContent = chunkTypeLabel(item.chunk_type);

            prevBtn.disabled = pos <= 0;
            nextBtn.disabled = pos >= citations.length - 1;
            const hideNav = citations.length <= 1;
            prevBtn.hidden = hideNav;
            nextBtn.hidden = hideNav;

            tabsRow.replaceChildren();
            if (citations.length > 1) {
                tabsRow.hidden = false;
                for (let idx = 0; idx < citations.length; idx++) {
                    const tabItem = citations[idx];
                    const tabBtn = document.createElement("button");
                    tabBtn.type = "button";
                    tabBtn.className = "citation-drawer-tab";
                    if (idx === pos) {
                        tabBtn.className = "citation-drawer-tab active";
                    }
                    if (tabItem.label) {
                        tabBtn.textContent = tabItem.label;
                    } else if (tabItem.index) {
                        tabBtn.textContent = "[Nguồn " + tabItem.index + "]";
                    } else {
                        tabBtn.textContent = "Nguồn " + (idx + 1);
                    }
                    tabBtn.addEventListener("click", () => {
                        citationDrawerState.activePos = idx;
                        renderActiveCitation();
                    });
                    tabsRow.appendChild(tabBtn);
                }
            } else {
                tabsRow.hidden = true;
            }

            metaBox.replaceChildren();
            const metaRows = [];
            let docTitle = item.document_name;
            if (!docTitle && item.source) {
                docTitle = String(item.source).split("\\").join("/").split("/").pop();
            }
            if (docTitle) {
                metaRows.push(["Tài liệu", docTitle]);
            }
            if (item.source) {
                metaRows.push(["Tệp nguồn", item.source]);
            }
            let sectionPath = item.breadcrumb;
            if (!sectionPath) {
                sectionPath = item.section_title;
            } else if (item.section_title && item.breadcrumb.indexOf(item.section_title) < 0) {
                sectionPath = item.breadcrumb + " > " + item.section_title;
            }
            if (sectionPath) {
                metaRows.push(["Mục / Phân cấp", sectionPath]);
            }
            if (Array.isArray(item.page_numbers) && item.page_numbers.length > 0) {
                metaRows.push(["Trang", "Trang " + item.page_numbers.join(", ")]);
            }
            for (const pair of metaRows) {
                const row = document.createElement("div");
                row.className = "citation-meta-row";
                const lbl = document.createElement("span");
                lbl.className = "citation-meta-label";
                lbl.textContent = pair[0];
                const val = document.createElement("span");
                val.className = "citation-meta-value";
                val.textContent = pair[1];
                row.append(lbl, val);
                metaBox.appendChild(row);
            }

            actionsBox.replaceChildren();
            const primaryPdfUrl = buildPdfPageUrl(item, null);
            if (primaryPdfUrl) {
                const openPdfLink = document.createElement("a");
                openPdfLink.className = "citation-open-pdf-btn";
                openPdfLink.href = primaryPdfUrl;
                openPdfLink.target = "_blank";
                openPdfLink.rel = "noopener noreferrer";
                if (Array.isArray(item.page_numbers) && item.page_numbers.length > 0) {
                    openPdfLink.textContent = "Mở file PDF gốc tại trang " + item.page_numbers[0] + " ↗";
                } else {
                    openPdfLink.textContent = "Mở file PDF gốc ↗";
                }
                actionsBox.appendChild(openPdfLink);

                if (Array.isArray(item.page_numbers) && item.page_numbers.length > 1) {
                    const jumpGroup = document.createElement("div");
                    jumpGroup.className = "citation-page-jump-group";
                    const jumpLabel = document.createElement("span");
                    jumpLabel.className = "citation-page-jump-label";
                    jumpLabel.textContent = "Nhảy trang:";
                    jumpGroup.appendChild(jumpLabel);
                    for (const pageNum of item.page_numbers) {
                        const pageUrl = buildPdfPageUrl(item, Number(pageNum));
                        if (!pageUrl) {
                            continue;
                        }
                        const jumpLink = document.createElement("a");
                        jumpLink.className = "citation-page-jump-btn";
                        jumpLink.href = pageUrl;
                        jumpLink.target = "_blank";
                        jumpLink.rel = "noopener noreferrer";
                        jumpLink.textContent = "Tr. " + pageNum;
                        jumpGroup.appendChild(jumpLink);
                    }
                    actionsBox.appendChild(jumpGroup);
                }
                actionsBox.hidden = false;
            } else {
                actionsBox.hidden = true;
            }

            excerptEl.replaceChildren();
            const excerptText = String(item.excerpt || "").trim();
            if (excerptText) {
                excerptEl.innerHTML = parseMarkdown(excerptText);
                copyBtn.hidden = false;
                copyBtn.textContent = "Sao chép";
            } else {
                const emptyNote = document.createElement("p");
                emptyNote.className = "citation-empty-note";
                emptyNote.textContent = "Không có đoạn trích văn bản lưu kèm cho nguồn này.";
                excerptEl.appendChild(emptyNote);
                copyBtn.hidden = true;
            }

            imagesBox.replaceChildren();
            if (Array.isArray(item.images) && item.images.length > 0) {
                const imgGallery = buildEvidenceImagesElement(item.images);
                if (imgGallery) {
                    imagesBox.appendChild(imgGallery);
                    imagesBox.hidden = false;
                } else {
                    imagesBox.hidden = true;
                }
            } else {
                imagesBox.hidden = true;
            }

            contentArea.scrollTop = 0;
        }

        prevBtn.addEventListener("click", () => {
            if (citationDrawerState.activePos > 0) {
                citationDrawerState.activePos = citationDrawerState.activePos - 1;
                renderActiveCitation();
            }
        });
        nextBtn.addEventListener("click", () => {
            if (citationDrawerState.activePos < citationDrawerState.citations.length - 1) {
                citationDrawerState.activePos = citationDrawerState.activePos + 1;
                renderActiveCitation();
            }
        });
        closeBtn.addEventListener("click", closeDrawer);
        backdrop.addEventListener("click", (event) => {
            if (event.target === backdrop) {
                closeDrawer();
            }
        });
        copyBtn.addEventListener("click", () => {
            const active = citationDrawerState.citations[citationDrawerState.activePos];
            if (!active || !active.excerpt) {
                return;
            }
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(active.excerpt).then(() => {
                    copyBtn.textContent = "Đã sao chép";
                    setTimeout(() => {
                        copyBtn.textContent = "Sao chép";
                    }, 1800);
                }).catch(() => {});
            }
        });

        document.addEventListener("keydown", (event) => {
            if (event.key !== "Escape") {
                return;
            }
            if (lightboxElements && !lightboxElements.overlay.hidden) {
                return;
            }
            if (!backdrop.hidden) {
                closeDrawer();
            }
        });

        citationDrawerElements = { backdrop, renderActiveCitation };
        return citationDrawerElements;
    }

    function openCitationDrawer(citationsList, activePos) {
        if (!Array.isArray(citationsList) || citationsList.length === 0) {
            return;
        }
        const drawer = ensureCitationDrawer();
        let pos = activePos;
        if (typeof pos !== "number" || pos < 0 || pos >= citationsList.length) {
            pos = 0;
        }
        citationDrawerState = { citations: citationsList, activePos: pos };
        drawer.renderActiveCitation();
        drawer.backdrop.hidden = false;
    }

    function buildSourcesCardElement(citationsList, fallbackSources) {
        if (citationsList.length === 0 && (!Array.isArray(fallbackSources) || fallbackSources.length === 0)) {
            return null;
        }
        const card = document.createElement("div");
        card.className = "sources-card";

        const header = document.createElement("div");
        header.className = "sources-header";
        if (citationsList.length > 0) {
            header.textContent = "Tài liệu đã trích dẫn (nhấn để xem đoạn gốc)";
        } else {
            header.textContent = "Tài liệu đã trích dẫn";
        }
        card.appendChild(header);

        const pills = document.createElement("div");
        pills.className = "sources-pills";

        if (citationsList.length > 0) {
            for (let pos = 0; pos < citationsList.length; pos++) {
                const cit = citationsList[pos];
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "source-tag source-tag-clickable";

                let docDisplay = cit.document_name;
                if (!docDisplay && cit.source) {
                    docDisplay = String(cit.source).split("\\").join("/").split("/").pop();
                }
                if (!docDisplay) {
                    docDisplay = "Tài liệu";
                }
                if (docDisplay.length > 46) {
                    docDisplay = docDisplay.slice(0, 43) + "...";
                }
                let labelPrefix = "";
                if (cit.label) {
                    labelPrefix = cit.label + " ";
                } else if (cit.index) {
                    labelPrefix = "[Nguồn " + cit.index + "] ";
                }
                let pageSuffix = "";
                if (Array.isArray(cit.page_numbers) && cit.page_numbers.length > 0) {
                    pageSuffix = " · Tr. " + cit.page_numbers.join(", ");
                }
                btn.textContent = labelPrefix + docDisplay + pageSuffix;
                btn.title = "Nhấn để xem chi tiết " + labelPrefix + docDisplay;
                btn.addEventListener("click", () => {
                    openCitationDrawer(citationsList, pos);
                });
                pills.appendChild(btn);
            }
        } else {
            const uniqueSources = Array.from(new Set(fallbackSources)).slice(0, 6);
            for (const src of uniqueSources) {
                const span = document.createElement("span");
                span.className = "source-tag";
                span.textContent = String(src);
                pills.appendChild(span);
            }
        }

        card.appendChild(pills);
        return card;
    }

    function openTraceInInspector(traceId) {
        if (traceId && window.ABankDebugInspector) {
            window.ABankDebugInspector.openTrace(String(traceId));
        }
    }

    function appendMessage(sender, content, meta = null) {
        exitWelcomeState();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + sender + "-message";

        if (sender === "user") {
            const bubbleEl = document.createElement("div");
            bubbleEl.className = "bubble";

            if (meta && Array.isArray(meta.userAttachments) && meta.userAttachments.length > 0) {
                const thumbsGrid = document.createElement("div");
                thumbsGrid.className = "user-attachments-grid";
                const lightboxItems = [];
                for (let idx = 0; idx < meta.userAttachments.length; idx++) {
                    const att = meta.userAttachments[idx];
                    if (!att || !att.data_url) {
                        continue;
                    }
                    let attName = "Ảnh đính kèm " + (idx + 1);
                    if (att.name) {
                        attName = att.name;
                    }
                    lightboxItems.push({
                        url: att.data_url,
                        caption: attName,
                        region_type: "attachment"
                    });
                }
                for (let idx = 0; idx < lightboxItems.length; idx++) {
                    const lbItem = lightboxItems[idx];
                    const thumbBtn = document.createElement("button");
                    thumbBtn.type = "button";
                    thumbBtn.className = "user-attachment-thumb";
                    thumbBtn.title = "Phóng to " + lbItem.caption;
                    const thumbImg = document.createElement("img");
                    thumbImg.src = lbItem.url;
                    thumbImg.alt = lbItem.caption;
                    thumbBtn.appendChild(thumbImg);
                    thumbBtn.addEventListener("click", () => {
                        openImageLightbox(lightboxItems, idx);
                    });
                    thumbsGrid.appendChild(thumbBtn);
                }
                if (thumbsGrid.childElementCount > 0) {
                    bubbleEl.appendChild(thumbsGrid);
                }
            }

            if (content) {
                const textPara = document.createElement("p");
                textPara.textContent = content;
                bubbleEl.appendChild(textPara);
            }
            msgDiv.appendChild(bubbleEl);
        } else {
            const citationList = [];
            const citationByIndex = new Map();
            const fallbackSources = [];
            if (meta) {
                for (const citation of meta.citations || []) {
                    if (typeof citation === "string") {
                        fallbackSources.push(citation);
                    } else if (citation && (citation.source || citation.index || citation.label)) {
                        citationList.push(citation);
                        if (typeof citation.index === "number") {
                            citationByIndex.set(citation.index, citationList.length - 1);
                        }
                    }
                }
                if (citationList.length === 0) {
                    for (const src of meta.sources || []) {
                        fallbackSources.push(src);
                    }
                }
            }

            const bubbleEl = document.createElement("div");
            bubbleEl.className = "bubble";
            bubbleEl.innerHTML = parseMarkdown(content, citationByIndex);

            if (citationList.length > 0) {
                bubbleEl.addEventListener("click", (event) => {
                    const btn = event.target.closest(".inline-citation-btn");
                    if (!btn) {
                        return;
                    }
                    const citeIdx = Number(btn.getAttribute("data-cite-idx"));
                    if (citationByIndex.has(citeIdx)) {
                        openCitationDrawer(citationList, citationByIndex.get(citeIdx));
                    }
                });
            }

            if (meta && Array.isArray(meta.images) && meta.images.length > 0) {
                const galleryEl = buildEvidenceImagesElement(meta.images);
                if (galleryEl) {
                    bubbleEl.appendChild(galleryEl);
                }
            }

            const sourcesCardEl = buildSourcesCardElement(citationList, fallbackSources);
            if (sourcesCardEl) {
                bubbleEl.appendChild(sourcesCardEl);
            }

            msgDiv.appendChild(bubbleEl);
        }

        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function setBusy(value) {
        busy = value;
        const locked = busy || Boolean(pendingCard);
        userInput.disabled = locked;
        sendBtn.disabled = locked;
        if (attachImageBtn) {
            attachImageBtn.disabled = locked;
        }
        if (voiceInputBtn) {
            voiceInputBtn.disabled = locked;
        }
        if (locked && isRecording) {
            stopVoiceRecording();
        }
        if (clearChatBtn) {
            clearChatBtn.disabled = busy;
        }
    }

    function createCardSvg(className, paths) {
        const ns = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(ns, "svg");
        svg.setAttribute("class", className);
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("fill", "none");
        svg.setAttribute("stroke", "currentColor");
        svg.setAttribute("stroke-width", "1.9");
        svg.setAttribute("stroke-linecap", "round");
        svg.setAttribute("stroke-linejoin", "round");
        svg.setAttribute("aria-hidden", "true");
        for (const item of paths) {
            const node = document.createElementNS(ns, item.tag);
            for (const [key, val] of Object.entries(item.attrs)) {
                node.setAttribute(key, val);
            }
            svg.appendChild(node);
        }
        return svg;
    }

    function renderInteraction(data) {
        const card = data.interaction;
        if (pendingCard && pendingCard.id === card.id) {
            return;
        }
        if (pendingCard) {
            pendingCard.element.remove();
        }
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }
        renderPlan(data.plan, false);
        finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
        exitWelcomeState();

        const isMulti = card.multi_select === true;
        const element = document.createElement("div");
        element.className = "message bot-message bot-message-interaction";

        const bubble = document.createElement("div");
        bubble.className = "bubble interaction-card";

        const header = document.createElement("div");
        header.className = "interaction-header";
        const headerIcon = createCardSvg("interaction-header-icon", [
            {
                tag: "path",
                attrs: {
                    d: "M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"
                }
            }
        ]);
        const title = document.createElement("h3");
        title.className = "interaction-title";
        title.id = "interaction-title-" + card.id;
        title.textContent = card.message;
        header.append(headerIcon, title);
        bubble.appendChild(header);

        const subtitle = document.createElement("p");
        subtitle.className = "interaction-subtitle";
        if (card.kind === "confirm") {
            subtitle.textContent = "Vui lòng xem phạm vi và xác nhận phương án bên dưới.";
        } else if (isMulti) {
            subtitle.textContent = "Vui lòng chọn một hoặc nhiều phương án bên dưới.";
        } else {
            subtitle.textContent = "Vui lòng chọn một trong các phương án bên dưới.";
        }
        bubble.appendChild(subtitle);

        if (Array.isArray(card.scope) && card.scope.length) {
            const scopeList = document.createElement("ol");
            scopeList.className = "interaction-scope";
            for (const item of card.scope) {
                const entry = document.createElement("li");
                entry.textContent = item.task;
                scopeList.appendChild(entry);
            }
            bubble.appendChild(scopeList);
        }

        const form = document.createElement("form");
        form.className = "interaction-form";
        form.setAttribute("aria-labelledby", title.id);
        form.noValidate = true;

        const fieldset = document.createElement("fieldset");
        fieldset.className = "interaction-fieldset";
        const legend = document.createElement("legend");
        legend.className = "visually-hidden";
        legend.textContent = "Danh sách lựa chọn";
        fieldset.appendChild(legend);

        const optionRows = [];
        let cancelOption = null;
        let hasOtherOption = false;

        for (const option of card.options) {
            if (option.id === "cancel") {
                cancelOption = option;
                continue;
            }
            if (option.id === "other") {
                hasOtherOption = true;
            }
            const inputId = "opt-" + card.id + "-" + option.id;
            const row = document.createElement("label");
            row.className = "interaction-option-row";
            row.htmlFor = inputId;

            const control = document.createElement("input");
            if (isMulti) {
                control.type = "checkbox";
            } else {
                control.type = "radio";
            }
            control.id = inputId;
            control.name = "interaction-choice-" + card.id;
            control.value = option.id;
            control.className = "interaction-control";

            const textSpan = document.createElement("span");
            textSpan.className = "interaction-option-text";
            textSpan.textContent = option.label;

            row.append(control, textSpan);
            fieldset.appendChild(row);
            optionRows.push({ option, row, control });
        }
        form.appendChild(fieldset);

        let otherWrap = null;
        let otherInput = null;
        if (hasOtherOption) {
            otherWrap = document.createElement("div");
            otherWrap.className = "interaction-other-wrap";
            otherWrap.hidden = true;

            const otherId = "other-" + card.id;
            const otherLabel = document.createElement("label");
            otherLabel.className = "visually-hidden";
            otherLabel.htmlFor = otherId;
            otherLabel.textContent = "Nhập lựa chọn khác";

            otherInput = document.createElement("input");
            otherInput.type = "text";
            otherInput.id = otherId;
            otherInput.className = "interaction-other-input";
            otherInput.maxLength = 500;
            otherInput.placeholder = "Nhập lựa chọn của bạn...";
            otherInput.addEventListener("keydown", event => {
                if (event.key === "Enter" && event.isComposing) {
                    event.preventDefault();
                }
            });

            otherWrap.append(otherLabel, otherInput);
            form.appendChild(otherWrap);
        }

        const divider = document.createElement("hr");
        divider.className = "interaction-divider";
        form.appendChild(divider);

        const summary = document.createElement("div");
        summary.className = "interaction-summary";
        summary.setAttribute("aria-live", "polite");
        summary.hidden = true;

        const summaryHead = document.createElement("div");
        summaryHead.className = "interaction-summary-head";
        const summaryIcon = createCardSvg("interaction-summary-icon", [
            { tag: "circle", attrs: { cx: "12", cy: "12", r: "9" } },
            { tag: "polyline", attrs: { points: "9 12 11 14 15 10" } }
        ]);
        const summaryTitle = document.createElement("strong");
        summaryHead.append(summaryIcon, summaryTitle);

        const summaryNote = document.createElement("p");
        summaryNote.className = "interaction-summary-note";
        summary.append(summaryHead, summaryNote);
        form.appendChild(summary);

        const footer = document.createElement("div");
        footer.className = "interaction-footer";

        const submitBtn = document.createElement("button");
        submitBtn.type = "submit";
        submitBtn.className = "interaction-submit-pill";
        submitBtn.textContent = "Xác nhận lựa chọn";
        footer.appendChild(submitBtn);

        const feedback = document.createElement("p");
        feedback.setAttribute("role", "status");

        if (cancelOption) {
            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "interaction-cancel-btn";
            cancelBtn.textContent = cancelOption.label;
            cancelBtn.addEventListener("click", () => {
                let payload;
                if (isMulti) {
                    payload = { interaction_id: card.id, choice_ids: ["cancel"] };
                } else {
                    payload = { interaction_id: card.id, choice_id: "cancel" };
                }
                submitInteractionReply(
                    card,
                    payload,
                    cancelOption.label,
                    false,
                    bubble,
                    feedback
                );
            });
            footer.appendChild(cancelBtn);
        }
        form.appendChild(footer);

        const syncSelectionState = () => {
            const selectedLabels = [];
            let otherChecked = false;
            for (const item of optionRows) {
                const isChecked = item.control.checked;
                item.row.classList.toggle("is-selected", isChecked);
                if (!isChecked) {
                    continue;
                }
                if (item.option.id === "other") {
                    otherChecked = true;
                    let customText = "";
                    if (otherInput) {
                        customText = otherInput.value.trim();
                    }
                    if (customText) {
                        selectedLabels.push("Khác: " + customText);
                    } else {
                        selectedLabels.push(item.option.label);
                    }
                } else {
                    selectedLabels.push(item.option.label);
                }
            }
            if (otherWrap && otherInput) {
                otherWrap.hidden = !otherChecked;
                if (!otherChecked) {
                    otherInput.removeAttribute("aria-invalid");
                }
            }
            if (selectedLabels.length === 0) {
                summary.hidden = true;
                summaryTitle.textContent = "";
                summaryNote.textContent = "";
            } else {
                summary.hidden = false;
                summaryTitle.textContent = "Đã chọn: " + selectedLabels.join("; ");
                if (otherChecked && otherInput && !otherInput.value.trim()) {
                    summaryNote.textContent = "Vui lòng nhập nội dung cho lựa chọn Khác.";
                } else if (isMulti && selectedLabels.length > 1) {
                    summaryNote.textContent = "Đã chọn " + selectedLabels.length + " phương án. Nhấn Xác nhận lựa chọn để tiếp tục.";
                } else {
                    summaryNote.textContent = "Nhấn Xác nhận lựa chọn để tiếp tục xử lý yêu cầu.";
                }
            }
            if (selectedLabels.length > 0 && (!otherChecked || (otherInput && otherInput.value.trim()))) {
                feedback.textContent = "";
            }
        };

        for (const item of optionRows) {
            item.control.addEventListener("change", () => {
                syncSelectionState();
                if (item.option.id === "other" && item.control.checked && otherInput) {
                    otherInput.focus();
                }
            });
        }
        if (otherInput) {
            otherInput.addEventListener("input", () => {
                if (otherInput.value.trim()) {
                    otherInput.removeAttribute("aria-invalid");
                }
                syncSelectionState();
            });
        }

        form.addEventListener("submit", event => {
            event.preventDefault();
            const selectedIds = [];
            const selectedLabels = [];
            for (const item of optionRows) {
                if (!item.control.checked) {
                    continue;
                }
                selectedIds.push(item.option.id);
                selectedLabels.push(item.option.label);
            }
            if (!selectedIds.length) {
                feedback.textContent = "Vui lòng chọn ít nhất một phương án bên dưới.";
                if (optionRows.length) {
                    optionRows[0].control.focus();
                }
                return;
            }
            const hasOther = selectedIds.includes("other");
            let customVal = "";
            if (otherInput) {
                customVal = otherInput.value.trim();
            }
            if (hasOther && !customVal) {
                feedback.textContent = "Vui lòng nhập nội dung cho lựa chọn Khác.";
                if (otherInput) {
                    otherInput.setAttribute("aria-invalid", "true");
                    otherInput.focus();
                }
                return;
            }
            if (otherInput) {
                otherInput.removeAttribute("aria-invalid");
            }
            const displayLabels = [];
            for (let idx = 0; idx < selectedIds.length; idx++) {
                if (selectedIds[idx] === "other") {
                    displayLabels.push(customVal);
                } else {
                    displayLabels.push(selectedLabels[idx]);
                }
            }
            let payload;
            if (isMulti) {
                payload = { interaction_id: card.id, choice_ids: selectedIds };
            } else {
                payload = { interaction_id: card.id, choice_id: selectedIds[0] };
            }
            if (hasOther) {
                payload.text = customVal;
            }
            const isOtherOnly = selectedIds.length === 1 && selectedIds[0] === "other";
            submitInteractionReply(
                card,
                payload,
                displayLabels.join("; "),
                isOtherOnly,
                bubble,
                feedback
            );
        });

        bubble.append(form, feedback);
        element.appendChild(bubble);
        pendingCard = { id: card.id, element };
        chatMessages.appendChild(element);
        setBusy(busy);
        scrollToBottom();
    }

    function renderAgentResult(data) {
        if (data && data.trace_id && window.ABankDebugInspector) {
            window.ABankDebugInspector.onTraceCompleted(data.trace_id);
        }
        if (data.status === "awaiting_input") {
            renderInteraction(data);
            return;
        }
        const isPassDone = Boolean(data && data.audit_status === "PASS");
        if (nodeStepper && Array.isArray(data.plan) && data.plan.length > 0) {
            nodeStepper.style.display = "block";
        }
        renderPlan(data.plan, isPassDone);
        finishProgressForStatus(data && data.audit_status, data && data.status);
        appendMessage("bot", data.answer || "Không nhận được phản hồi từ hệ thống.", data);
    }

    function postAgent(path, payload) {
        return fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: currentSessionId, ...payload })
        });
    }

    async function recoverInteraction(interactionId = null, restoreResult = false) {
        try {
            const response = await postAgent("/api/agent/status", {});
            if (!response.ok) {
                return false;
            }
            const data = await response.json();
            if (data.status === "awaiting_input") {
                renderInteraction(data);
                return data.interaction.id !== interactionId;
            }
            if (data.answer && (restoreResult || (interactionId && data.accepted_interaction_id === interactionId))) {
                if (pendingCard) {
                    pendingCard.element.remove();
                    pendingCard = null;
                }
                renderAgentResult(data);
                return true;
            }
        } catch {
            return false;
        }
        return false;
    }

    async function submitInteractionReply(card, payload, displayLabel, isOtherOnly, bubble, feedback) {
        if (busy || !pendingCard || pendingCard.id !== card.id) {
            return;
        }
        setBusy(true);
        const controls = bubble.querySelectorAll("button, input");
        controls.forEach(control => { control.disabled = true; });
        feedback.textContent = "Đang gửi lựa chọn…";
        if (nodeStepper) {
            nodeStepper.style.display = "block";
        }
        updateStepperProgress("Đang gửi lựa chọn và tiếp tục xử lý...", {
            kind: "interaction",
            name: "resume",
            status: "RUNNING"
        });
        try {
            const response = await postAgent("/api/agent/resume", payload);
            if (response.status === 422) {
                const data = await response.json();
                feedback.textContent = "Vui lòng kiểm tra lại lựa chọn.";
                if (typeof data.detail === "string") {
                    feedback.textContent = data.detail;
                }
                finishProgressForStatus("AWAITING_INPUT", "awaiting_input");
                return;
            }
            if (!response.ok) {
                throw new Error("Không nhận được kết quả tiếp tục.");
            }
            const data = await response.json();
            pendingCard = null;
            bubble.querySelectorAll(".interaction-subtitle, .interaction-scope, .interaction-form").forEach(group => group.remove());
            feedback.remove();
            if (isOtherOnly) {
                appendMessage("user", "Đã gửi câu trả lời khác.");
            } else {
                appendMessage("user", displayLabel);
            }
            renderAgentResult(data);
        } catch {
            if (!await recoverInteraction(card.id)) {
                feedback.textContent = "Chưa nhận được kết quả. Bạn có thể thử lại.";
            }
        } finally {
            if (pendingCard && pendingCard.id === card.id) {
                controls.forEach(control => { control.disabled = false; });
            }
            setBusy(false);
        }
    }

    async function sendChoice(card, choiceId, label, bubble, feedback) {
        const payload = { interaction_id: card.id, choice_id: choiceId };
        if (choiceId === "other") {
            payload.text = label;
        }
        return submitInteractionReply(card, payload, label, choiceId === "other", bubble, feedback);
    }

    function showComposerStatus(message, mode = "info", autoHideMs = 0) {
        if (!composerStatusBanner) {
            return;
        }
        if (statusBannerTimer) {
            clearTimeout(statusBannerTimer);
            statusBannerTimer = null;
        }
        composerStatusBanner.className = "composer-status-banner";
        if (mode === "recording") {
            composerStatusBanner.classList.add("is-recording");
        } else if (mode === "error") {
            composerStatusBanner.classList.add("is-error");
        }
        composerStatusBanner.textContent = message;
        composerStatusBanner.hidden = false;
        if (autoHideMs > 0) {
            statusBannerTimer = setTimeout(() => {
                clearComposerStatus();
            }, autoHideMs);
        }
    }

    function clearComposerStatus() {
        if (!composerStatusBanner) {
            return;
        }
        if (statusBannerTimer) {
            clearTimeout(statusBannerTimer);
            statusBannerTimer = null;
        }
        composerStatusBanner.textContent = "";
        composerStatusBanner.className = "composer-status-banner";
        composerStatusBanner.hidden = true;
    }

    function formatFileSize(bytes) {
        if (typeof bytes !== "number" || bytes <= 0) {
            return "";
        }
        if (bytes < 1024) {
            return bytes + " B";
        }
        if (bytes < 1024 * 1024) {
            return Math.round(bytes / 1024) + " KB";
        }
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function isAllowedImageFile(file) {
        if (!file) {
            return false;
        }
        const mime = String(file.type || "").toLowerCase();
        if (ALLOWED_IMAGE_TYPES.includes(mime)) {
            return true;
        }
        if (!mime && file.name) {
            const lowerName = String(file.name).toLowerCase();
            for (const ext of ALLOWED_IMAGE_EXTS) {
                if (lowerName.endsWith(ext)) {
                    return true;
                }
            }
        }
        return false;
    }

    function readFileAsDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve(String(reader.result || ""));
            };
            reader.onerror = () => {
                reject(new Error("Không thể đọc tệp hình ảnh."));
            };
            reader.readAsDataURL(file);
        });
    }

    function renderAttachmentPreviewBar() {
        if (!attachmentPreviewBar) {
            return;
        }
        attachmentPreviewBar.replaceChildren();
        if (pendingAttachments.length === 0) {
            attachmentPreviewBar.hidden = true;
            return;
        }
        attachmentPreviewBar.hidden = false;
        const previewLightboxItems = [];
        for (let idx = 0; idx < pendingAttachments.length; idx++) {
            const att = pendingAttachments[idx];
            previewLightboxItems.push({
                url: att.data_url,
                caption: att.name,
                region_type: "attachment"
            });
        }
        for (let idx = 0; idx < pendingAttachments.length; idx++) {
            const att = pendingAttachments[idx];
            const card = document.createElement("div");
            card.className = "attachment-thumb-card";

            const thumbBtn = document.createElement("button");
            thumbBtn.type = "button";
            thumbBtn.className = "attachment-thumb-btn";
            thumbBtn.title = "Xem trước " + att.name;
            const img = document.createElement("img");
            img.src = att.data_url;
            img.alt = att.name;
            thumbBtn.appendChild(img);
            thumbBtn.addEventListener("click", () => {
                openImageLightbox(previewLightboxItems, idx);
            });

            const meta = document.createElement("div");
            meta.className = "attachment-thumb-meta";
            const nameEl = document.createElement("span");
            nameEl.className = "attachment-thumb-name";
            nameEl.textContent = att.name;
            const sizeEl = document.createElement("span");
            sizeEl.className = "attachment-thumb-size";
            sizeEl.textContent = formatFileSize(att.size);
            meta.append(nameEl, sizeEl);

            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "attachment-remove-btn";
            removeBtn.textContent = "×";
            removeBtn.setAttribute("aria-label", "Xóa ảnh " + att.name);
            removeBtn.addEventListener("click", () => {
                pendingAttachments.splice(idx, 1);
                renderAttachmentPreviewBar();
            });

            card.append(thumbBtn, meta, removeBtn);
            attachmentPreviewBar.appendChild(card);
        }
    }

    async function addImageFiles(fileList) {
        if (!fileList || fileList.length === 0) {
            return;
        }
        for (let i = 0; i < fileList.length; i++) {
            const file = fileList[i];
            if (pendingAttachments.length >= MAX_ATTACHMENTS) {
                showComposerStatus("Chỉ đính kèm tối đa " + MAX_ATTACHMENTS + " ảnh cho mỗi lượt gửi.", "error", 4000);
                break;
            }
            if (!isAllowedImageFile(file)) {
                showComposerStatus("Định dạng không hỗ trợ (" + (file.name || "tệp") + "). Vui lòng chọn PNG, JPEG, WEBP hoặc GIF.", "error", 4500);
                continue;
            }
            if (file.size > MAX_IMAGE_BYTES) {
                showComposerStatus("Ảnh " + (file.name || "") + " vượt quá giới hạn 5 MB.", "error", 4500);
                continue;
            }
            try {
                const dataUrl = await readFileAsDataUrl(file);
                if (!dataUrl.startsWith("data:image/")) {
                    showComposerStatus("Không đọc được dữ liệu hình ảnh hợp lệ.", "error", 4000);
                    continue;
                }
                let displayName = file.name;
                if (!displayName) {
                    displayName = "anh-dinh-kem-" + (pendingAttachments.length + 1) + ".png";
                }
                pendingAttachments.push({
                    name: displayName,
                    size: file.size,
                    data_url: dataUrl
                });
            } catch {
                showComposerStatus("Không thể tải ảnh " + (file.name || "") + ".", "error", 4000);
            }
        }
        renderAttachmentPreviewBar();
    }

    function setVoiceRecordingUi(active) {
        isRecording = active;
        if (!voiceInputBtn) {
            return;
        }
        voiceInputBtn.classList.toggle("is-recording", active);
        if (active) {
            voiceInputBtn.setAttribute("aria-pressed", "true");
            voiceInputBtn.title = "Nhấn để dừng ghi âm";
        } else {
            voiceInputBtn.setAttribute("aria-pressed", "false");
            voiceInputBtn.title = "Nhập bằng giọng nói tiếng Việt";
        }
    }

    function stopVoiceRecording() {
        if (speechRecognition && isRecording) {
            try {
                speechRecognition.stop();
            } catch {
                setVoiceRecordingUi(false);
            }
        }
    }

    function initVoiceInput() {
        if (!voiceInputBtn) {
            return;
        }
        const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionCtor) {
            voiceInputBtn.addEventListener("click", () => {
                showComposerStatus("Trình duyệt hiện tại chưa hỗ trợ nhận diện giọng nói. Vui lòng dùng Chrome hoặc Edge.", "error", 5000);
            });
            return;
        }

        voiceInputBtn.addEventListener("click", () => {
            if (busy || pendingCard) {
                return;
            }
            if (isRecording && speechRecognition) {
                stopVoiceRecording();
                return;
            }

            const recognition = new SpeechRecognitionCtor();
            recognition.lang = "vi-VN";
            recognition.interimResults = true;
            recognition.continuous = false;
            recognition.maxAlternatives = 1;
            speechRecognition = recognition;
            voiceBaseText = userInput.value.trim();
            voiceCapturedText = "";

            recognition.onstart = () => {
                setVoiceRecordingUi(true);
                showComposerStatus("Đang nghe giọng nói tiếng Việt… Hãy nói câu hỏi của Quý khách.", "recording");
            };

            recognition.onresult = (event) => {
                let finalPart = "";
                let interimPart = "";
                for (let i = 0; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalPart = finalPart + transcript + " ";
                    } else {
                        interimPart = interimPart + transcript;
                    }
                }
                const combinedVoice = (finalPart + interimPart).trim();
                voiceCapturedText = combinedVoice;
                if (voiceBaseText && combinedVoice) {
                    userInput.value = voiceBaseText + " " + combinedVoice;
                } else if (combinedVoice) {
                    userInput.value = combinedVoice;
                } else {
                    userInput.value = voiceBaseText;
                }
            };

            recognition.onerror = (event) => {
                setVoiceRecordingUi(false);
                const errCode = String((event && event.error) || "");
                let errText = "Không thể nhận diện giọng nói lúc này.";
                if (errCode === "not-allowed" || errCode === "service-not-allowed") {
                    errText = "Chưa cấp quyền micro. Vui lòng cho phép truy cập micro trong trình duyệt.";
                } else if (errCode === "no-speech") {
                    errText = "Chưa phát hiện giọng nói. Quý khách vui lòng nhấn micro và thử lại.";
                } else if (errCode === "audio-capture") {
                    errText = "Không tìm thấy thiết bị micro khả dụng.";
                } else if (errCode === "network") {
                    errText = "Lỗi kết nối mạng khi nhận diện giọng nói.";
                }
                showComposerStatus(errText, "error", 5000);
            };

            recognition.onend = () => {
                const hadSpeech = Boolean(voiceCapturedText);
                setVoiceRecordingUi(false);
                if (hadSpeech) {
                    showComposerStatus("Đã ghi nhận nội dung giọng nói vào ô nhập.", "info", 3000);
                    userInput.focus();
                } else if (composerStatusBanner && composerStatusBanner.classList.contains("is-recording")) {
                    clearComposerStatus();
                }
            };

            try {
                recognition.start();
            } catch {
                setVoiceRecordingUi(false);
                showComposerStatus("Không thể khởi động micro. Vui lòng thử lại.", "error", 4000);
            }
        });
    }

    async function submitQueryFallback(cleanQuery, attachmentsPayload) {
        try {
            const reqBody = {
                message: cleanQuery,
                session_id: currentSessionId
            };
            if (Array.isArray(attachmentsPayload) && attachmentsPayload.length > 0) {
                reqBody.attachments = attachmentsPayload;
            }
            const res = await fetch("/api/agent/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reqBody)
            });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: "Lỗi kết nối máy chủ dịch vụ" }));
                let detailMsg = "Không thể xử lý yêu cầu lúc này.";
                if (errData && errData.detail) {
                    detailMsg = errData.detail;
                }
                finishProgressForStatus("ERROR", "error");
                appendMessage("bot", "**Rất tiếc:** " + detailMsg);
            } else {
                const data = await res.json();
                renderAgentResult(data);
            }
        } catch (err) {
            finishProgressForStatus("ERROR", "error");
            appendMessage("bot", "**Lỗi kết nối:** Không thể liên lạc với máy chủ (" + err.message + ").");
        }
    }

    async function submitQuery(queryText) {
        if (busy || pendingCard) {
            return;
        }
        let cleanQuery = "";
        if (queryText) {
            cleanQuery = queryText.trim();
        }
        const attachmentsToSend = pendingAttachments.slice();
        if (cleanQuery === "" && attachmentsToSend.length === 0) {
            return;
        }

        if (isRecording) {
            stopVoiceRecording();
        }
        clearComposerStatus();

        const attachmentsPayload = [];
        for (const att of attachmentsToSend) {
            attachmentsPayload.push({ name: att.name, data_url: att.data_url });
        }

        let userBubbleText = cleanQuery;
        if (!userBubbleText && attachmentsToSend.length > 0) {
            userBubbleText = "Đã gửi " + attachmentsToSend.length + " hình ảnh đính kèm để phân tích.";
        }

        exitWelcomeState();
        appendMessage("user", userBubbleText, { userAttachments: attachmentsToSend });
        userInput.value = "";
        pendingAttachments = [];
        renderAttachmentPreviewBar();
        setBusy(true);

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
        let activeStreamTraceId = null;

        try {
            const streamBody = {
                message: cleanQuery,
                session_id: currentSessionId
            };
            if (attachmentsPayload.length > 0) {
                streamBody.attachments = attachmentsPayload;
            }
            const response = await fetch("/api/agent/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(streamBody)
            });

            if (!response.ok || !response.body) {
                botMsgDiv.remove();
                if (!await recoverInteraction() && response.status < 500) {
                    await submitQueryFallback(cleanQuery, attachmentsPayload);
                } else if (!pendingCard) {
                    finishProgressForStatus("ERROR", "error");
                    appendMessage("bot", "Không nhận được phản hồi từ hệ thống.");
                }
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
                                if (eventData.trace_id && eventData.trace_id !== activeStreamTraceId) {
                                    activeStreamTraceId = eventData.trace_id;
                                    if (window.ABankDebugInspector) {
                                        window.ABankDebugInspector.onLiveTraceStart(activeStreamTraceId);
                                    }
                                }
                                updateStepperProgress(eventData.description, eventData.action);
                                renderPlan(eventData.plan, false);
                            } else if (eventType === "awaiting_input") {
                                finalDoneMeta = eventData;
                            } else if (eventType === "token") {
                                accumulatedText = accumulatedText + eventData.content;
                                if (streamingBubble) {
                                    streamingBubble.innerHTML = parseMarkdown(accumulatedText);
                                    scrollToBottom();
                                }
                            } else if (eventType === "done") {
                                finalDoneMeta = eventData;
                                const isPassDone = eventData.audit_status === "PASS";
                                renderPlan(eventData.plan, isPassDone);
                                finishProgressForStatus(eventData.audit_status, eventData.status);
                            } else if (eventType === "error") {
                                finalDoneMeta = eventData;
                                finishProgressForStatus("ERROR", "error");
                            }
                        } catch (e) {
                            console.warn("Parse SSE event error:", e);
                        }
                    }
                }
            }

            if (finalDoneMeta) {
                botMsgDiv.remove();
                if (finalDoneMeta.event_type === "error") {
                    let errMsg = "Đã xảy ra lỗi khi xử lý yêu cầu.";
                    if (finalDoneMeta.message) {
                        errMsg = finalDoneMeta.message;
                    }
                    finishProgressForStatus("ERROR", "error");
                    appendMessage("bot", "**Rất tiếc:** " + errMsg, {
                        trace_id: finalDoneMeta.trace_id,
                        audit_status: "ERROR"
                    });
                    if (window.ABankDebugInspector) {
                        window.ABankDebugInspector.onTraceCompleted(finalDoneMeta.trace_id || activeStreamTraceId);
                    }
                    return;
                }
                if (finalDoneMeta.status === "awaiting_input") {
                    if (window.ABankDebugInspector) {
                        window.ABankDebugInspector.onTraceCompleted(finalDoneMeta.trace_id || activeStreamTraceId);
                    }
                    renderInteraction(finalDoneMeta);
                    return;
                }
                let finalText = accumulatedText;
                if (finalDoneMeta.full_answer) {
                    finalText = finalDoneMeta.full_answer;
                }

                appendMessage("bot", finalText, {
                    citations: finalDoneMeta.citations,
                    sources: finalDoneMeta.sources,
                    images: finalDoneMeta.images,
                    trace_id: finalDoneMeta.trace_id,
                    turn_id: finalDoneMeta.turn_id,
                    audit_status: finalDoneMeta.audit_status,
                    latency_ms: finalDoneMeta.latency_ms
                });
                if (window.ABankDebugInspector) {
                    window.ABankDebugInspector.onTraceCompleted(finalDoneMeta.trace_id || activeStreamTraceId);
                }
            } else {
                if (window.ABankDebugInspector && activeStreamTraceId) {
                    window.ABankDebugInspector.onTraceCompleted(activeStreamTraceId);
                }
                botMsgDiv.remove();
                if (!await recoverInteraction()) {
                    finishProgress("Kết nối bị gián đoạn");
                    appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
                }
            }

        } catch (err) {
            botMsgDiv.remove();
            if (await recoverInteraction()) {
                return;
            }
            if (streamStarted) {
                finishProgress("Kết nối bị gián đoạn");
                appendMessage("bot", "Kết nối bị gián đoạn trước khi nhận được phản hồi.");
            } else {
                await submitQueryFallback(cleanQuery, attachmentsPayload);
            }
        } finally {
            setBusy(false);
            userInput.focus();
            scrollToBottom();
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery(userInput.value);
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.isComposing) {
            e.preventDefault();
        }
    });

    userInput.addEventListener("paste", (e) => {
        if (busy || pendingCard || !e.clipboardData || !e.clipboardData.items) {
            return;
        }
        const pastedImages = [];
        for (const item of e.clipboardData.items) {
            if (item.kind === "file" && item.type && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                    pastedImages.push(file);
                }
            }
        }
        if (pastedImages.length > 0) {
            e.preventDefault();
            addImageFiles(pastedImages);
        }
    });

    if (attachImageBtn && imageUploadInput) {
        attachImageBtn.addEventListener("click", () => {
            if (busy || pendingCard) {
                return;
            }
            imageUploadInput.click();
        });
        imageUploadInput.addEventListener("change", () => {
            if (imageUploadInput.files && imageUploadInput.files.length > 0) {
                addImageFiles(imageUploadInput.files);
            }
            imageUploadInput.value = "";
        });
    }

    if (chatComposer) {
        chatComposer.addEventListener("dragover", (e) => {
            e.preventDefault();
            if (!busy && !pendingCard) {
                chatComposer.classList.add("is-dragover");
            }
        });
        chatComposer.addEventListener("dragleave", () => {
            chatComposer.classList.remove("is-dragover");
        });
        chatComposer.addEventListener("drop", (e) => {
            e.preventDefault();
            chatComposer.classList.remove("is-dragover");
            if (busy || pendingCard) {
                return;
            }
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                addImageFiles(e.dataTransfer.files);
            }
        });
    }

    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => {
            currentSessionId = newSessionId();
            sessionStorage.setItem(sessionKey, currentSessionId);
            pendingCard = null;
            pendingAttachments = [];
            renderAttachmentPreviewBar();
            stopVoiceRecording();
            clearComposerStatus();
            setBusy(false);
            chatMessages.innerHTML = "";
            enterWelcomeState();
            if (nodeStepper) {
                nodeStepper.style.display = "none";
            }
            if (window.ABankDebugInspector) {
                window.ABankDebugInspector.setSessionId(currentSessionId);
            }
            userInput.value = "";
            userInput.focus();
        });
    }

    initVoiceInput();

    setBusy(true);
    recoverInteraction(null, true).finally(() => {
        setBusy(false);
        userInput.focus();
    });
});
