document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const messagesContainer = document.getElementById("messages-container");
    const clearBtn = document.getElementById("clear-btn");
    const statusDb = document.getElementById("status-db");
    const statusGemini = document.getElementById("status-gemini");
    const suggestButtons = document.querySelectorAll(".suggest-btn");

    // Check System Integrity on load
    async function checkSystemIntegrity() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            
            if (data.status === "healthy") {
                statusDb.className = "status-indicator online";
                statusDb.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connected';
                statusGemini.className = "status-indicator online";
                statusGemini.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connected';
            } else {
                if (data.has_pinecone_key && !data.db_error) {
                    statusDb.className = "status-indicator online";
                    statusDb.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connected';
                } else {
                    statusDb.className = "status-indicator offline";
                    statusDb.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Error';
                }
                
                if (data.has_gemini_key) {
                    statusGemini.className = "status-indicator online";
                    statusGemini.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connected';
                } else {
                    statusGemini.className = "status-indicator offline";
                    statusGemini.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Error';
                }
            }
        } catch (error) {
            console.error("Health check failed:", error);
            statusDb.className = "status-indicator offline";
            statusDb.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Offline';
            statusGemini.className = "status-indicator offline";
            statusGemini.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Offline';
        }
    }

    checkSystemIntegrity();

    // Suggestion Buttons click handler
    suggestButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            userInput.value = btn.innerText;
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    // Handle Form Submit
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Clear input field
        userInput.value = "";

        // Append User Message
        appendMessage(text, "user");

        // Show Typing Indicator
        const typingIndicator = showTypingIndicator();

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });

            // Remove Typing Indicator
            typingIndicator.remove();

            if (!res.ok) {
                const errData = await res.json();
                appendMessage(`Error: ${errData.detail || "Something went wrong."}`, "bot", { isError: true });
                return;
            }

            const data = await res.json();
            appendMessage(data.answer, "bot", {
                source: data.source,
                docs: data.docs,
                latency: data.latency_ms
            });

        } catch (error) {
            console.error("Chat API failed:", error);
            typingIndicator.remove();
            appendMessage("Failed to communicate with the server. Please ensure the backend is running.", "bot", { isError: true });
        }
    });

    // Helper: Append message bubbles
    function appendMessage(text, role, meta = {}) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;

        const msgContent = document.createElement("div");
        msgContent.className = "message-content";
        
        if (meta.isError) {
            msgContent.style.borderColor = "hsl(0, 76%, 50%)";
            msgContent.style.color = "hsl(0, 76%, 70%)";
        }

        // Text Content
        const textParagraph = document.createElement("p");
        textParagraph.innerText = text;
        msgContent.appendChild(textParagraph);

        // Meta (latency, source)
        if (role === "bot" && !meta.isError && meta.source) {
            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";

            const isDoc = meta.source === "document";
            const pillSpan = document.createElement("span");
            pillSpan.className = `source-pill ${isDoc ? 'doc' : 'general'}`;
            
            if (isDoc) {
                pillSpan.innerHTML = '<i class="fa-solid fa-file-invoice"></i> From document';
            } else {
                pillSpan.innerHTML = '<i class="fa-solid fa-globe"></i> General knowledge';
            }

            const latencySpan = document.createElement("span");
            latencySpan.className = "latency";
            latencySpan.innerText = `${meta.latency}ms`;

            metaDiv.appendChild(pillSpan);
            metaDiv.appendChild(latencySpan);
            msgContent.appendChild(metaDiv);

            // Document citations (Accordion)
            if (isDoc && meta.docs && meta.docs.length > 0) {
                const citationDiv = document.createElement("div");
                citationDiv.className = "citation-details";

                const accTrigger = document.createElement("button");
                accTrigger.className = "accordion-trigger";
                accTrigger.innerHTML = '<i class="fa-solid fa-chevron-down"></i> View Retrieved Chunks';

                const accContent = document.createElement("div");
                accContent.className = "accordion-content";

                meta.docs.forEach((doc, idx) => {
                    const chunkDiv = document.createElement("div");
                    chunkDiv.className = "citation-chunk";

                    const chunkText = document.createElement("div");
                    chunkText.className = "chunk-text";
                    chunkText.innerText = doc.content.trim();

                    const chunkSource = document.createElement("div");
                    chunkSource.className = "chunk-source";
                    
                    const docName = doc.metadata.source ? doc.metadata.source.split(/[/\\]/).pop() : "Unknown Doc";
                    const docPage = doc.metadata.page !== undefined ? `Page ${doc.metadata.page + 1}` : "";
                    
                    chunkSource.innerHTML = `<span>Source: ${docName}</span><span>${docPage}</span>`;

                    chunkDiv.appendChild(chunkText);
                    chunkDiv.appendChild(chunkSource);
                    accContent.appendChild(chunkDiv);
                });

                accTrigger.addEventListener("click", () => {
                    accTrigger.classList.toggle("active");
                    accContent.classList.toggle("open");
                    if (accContent.classList.contains("open")) {
                        accContent.style.maxHeight = accContent.scrollHeight + "px";
                    } else {
                        accContent.style.maxHeight = "0";
                    }
                });

                citationDiv.appendChild(accTrigger);
                citationDiv.appendChild(accContent);
                msgContent.appendChild(citationDiv);
            }
        }

        msgDiv.appendChild(msgContent);
        messagesContainer.appendChild(msgDiv);
        
        // Auto scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Helper: Show typing indicator
    function showTypingIndicator() {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot typing-bubble";

        const msgContent = document.createElement("div");
        msgContent.className = "message-content";

        const indicator = document.createElement("div");
        indicator.className = "typing-indicator";
        indicator.innerHTML = "<span></span><span></span><span></span>";

        msgContent.appendChild(indicator);
        msgDiv.appendChild(msgContent);
        messagesContainer.appendChild(msgDiv);

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return msgDiv;
    }

    // Clear Chat Button
    clearBtn.addEventListener("click", () => {
        // Remove all user and bot messages except the first system message
        const messages = Array.from(messagesContainer.children);
        messages.forEach(msg => {
            if (!msg.classList.contains("system")) {
                msg.remove();
            }
        });
    });
});
