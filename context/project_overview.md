# **Architectural Framework for Automated Multi-Agent Video Production Using Google ADK**

The convergence of autonomous agentic workflows, generative multimodal language models, and programmatic media synthesis has fundamentally altered the landscape of digital video engineering. The traditional paradigm of video editing—characterized by manual media ingestion, subjective human curation, rigid non-linear editor (NLE) timelines, and labor-intensive post-production—is rapidly being supplanted by deterministic, artificial intelligence-driven pipelines. To architect a system capable of analyzing localized media, orchestrating a cohesive narrative, facilitating human-in-the-loop review, and rendering a highly engaging final artifact with advanced transitions requires a robust, scalable framework.  
This report delineates a comprehensive, end-to-end software architecture utilizing the Google Agent Development Kit (ADK) as the central orchestration engine. By strictly adhering to a six-stage workflow—local media ingestion, multimodal video understanding, autonomous internal AI review, human-in-the-loop editorial review, state finalization, and high-performance FFmpeg transition rendering—this framework establishes a fully automated, broadcast-quality video production pipeline tailored for modern platforms such as YouTube.

## **System Architecture and Orchestration Overview**

The core of this automated video pipeline relies on the Google Agent Development Kit (ADK). Introduced as an open-source, code-first agent development framework, the ADK is designed to build, debug, and deploy reliable artificial intelligence agents at an enterprise scale1. Unlike rudimentary prompt-chaining libraries that often fail when context windows overflow, the ADK treats agentic behavior as deterministic software engineering, providing native abstractions for memory management, external tool invocation, safety callbacks, and complex multi-agent routing2.  
To fulfill the requirements of a multi-stage video production workflow, the system architecture abandons the single-agent approach in favor of a hierarchical, multi-agent pipeline. The ADK facilitates this through its orchestration primitives, specifically the SequentialAgent and LoopAgent constructs, which force a strict execution order akin to a digital assembly line6. This ensures that the pipeline does not attempt to render a video before the human supervisor has reviewed the timeline.  
The proposed overarching pipeline consists of the following specialized sub-agents acting in sequence:

> 1. **Root Greeter Agent:** Initiates the session by asking the user for the day's theme or main highlight. Once provided, it stores this context in the session state and triggers the automated pipeline7.  
> 2. **Ingestion and Contextualization Agent:** Reads the user's localized directory containing the downloaded Google Photos clips, mapping file paths and extracting raw metadata.  
> 3. **Autonomous Editing Loop (LoopAgent):** A looped multi-agent workflow6 designed to iteratively perfect the video draft before human review. It contains:  
   * **Curation and Narrative Agent:** Generates the initial Edit Decision List (EDL) from the raw footage.  
   * **Reviewer Agent:** A master of the YouTube algorithm that strictly analyzes the EDL against viral metrics and analytics.  
   * **Critic Agent:** Evaluates the video from a human perspective, checking thumbnail viability, first 30-second retention (AVD), and Click-Through Rate (CTR) potential.  
> 4. **Review Orchestrator Agent:** Translates the approved EDL into an OpenTimelineIO (.otio) proxy, suspends the autonomous loop, and interfaces with the user via a web or streaming interface to present the strictly evaluated draft for approval.  
> 5. **Enhancement and Rendering Agent:** Upon approval, executes complex FFmpeg toolsets to bake in advanced visual transitions, audio ducking, and auto-generated captions, ultimately saving the final video.

These agents are bound together by a root workflow agent that passes session state—such as localized file paths, JSON-formatted EDLs, and human feedback parameters—across the boundaries of each specific task4.

## **Phase I: Intent Capture, Local Media Ingestion and File Contextualization**

Before processing any files, the system must understand the creator's editorial intent. The workflow begins with the Root Greeter Agent prompting the user (via the CLI or web UI) to describe the vlog's theme or the day's main highlight. Once the user responds, this thematic context is captured and saved within the ADK session state, acting as the narrative anchor for the rest of the pipeline4.  
The foundational step of the user's specified workflow then dictates that raw video clips are manually downloaded from Google Photos and stored within a designated local directory. The ADK framework must, therefore, be capable of autonomously scanning this directory, ingesting the file structures, and preparing the media for downstream multimodal analysis.

### **The Artifact Service and Local Directory Reading**

The Google ADK provides an internal mechanism known as the Artifact Service, which is designed to allow agents to securely read and write files within the host environment9. To initiate the workflow, the Ingestion Agent must utilize a custom-built Python tool—often implemented as a simple function wrapped in the @dataclass or standard ADK tool configuration—that interfaces with the local operating system10.  
When the pipeline is triggered, the Ingestion Agent invokes a scan\_local\_directory tool. This tool utilizes standard Python libraries (such as os and pathlib) to iterate through the target folder, filtering specifically for valid video file extensions (e.g., .mp4, .mov)12. The engineering challenge in this phase is not simply locating the files, but structuring the data in a manner that the subsequent Large Language Model (LLM) can easily comprehend.  
The Ingestion Agent constructs a comprehensive manifest of the local media, which is subsequently injected into the ADK's InMemorySessionService or passed directly to the next agent in the sequence9.  
Table 1 illustrates the structured data schema that the Ingestion Agent creates and stores within the ADK session state to map the localized Google Photos downloads:

| Metadata Field | Data Type | Architectural Purpose | Example Value |
| :---- | :---- | :---- | :---- |
| clip\_id | String (UUID) | Provides a unique identifier for downstream NLE tracking and transition mapping. | vid\_8f3a9b21 |
| absolute\_path | String | Instructs FFmpeg exactly where the localized file resides on the disk. | C:/Users/Media/Photos/clip\_01.mp4 |
| duration\_seconds | Float | Essential for calculating offsets during the transition rendering phase. | 14.35 |
| frame\_rate | Float | Required to ensure all clips are normalized before concatenation to prevent audio desynchronization. | 29.97 |
| resolution | String | Identifies heterogeneous media (e.g., mixing 4K and 1080p footage) requiring pre-scaling. | 1920x1080 |

By anchoring the workflow in a strictly defined local directory map, the ADK framework establishes a reliable ground truth. This bypasses the unreliability of ephemeral cloud URLs and ensures that the heavy computational rendering phases have immediate, unthrottled access to the raw media bytes.

## **Phase II: Multimodal Video Analysis and Initial Scene Selection**

Once the local file context is established, the application must "read the movie files and come up with the final video." A highly engaging YouTube video relies heavily on meticulous pacing, the removal of dead air, and the extraction of the most visually compelling moments. In this automated framework, the ADK Curation Agent takes the first pass at this by utilizing the Gemini Multimodal API.

### **Generative Video Understanding**

The Gemini 1.5 and 2.0 Flash models feature native, deep-level video understanding capabilities, allowing them to process sequences of frames and intertwined audio tracks directly without relying on intermediary transcription layers14. The Curation Agent iterates through the file paths stored in the ADK session state by the Ingestion Agent, utilizing the Google GenAI SDK to pass the local video files to the Gemini API as context14.  
The agent is provided with strict behavioral instructions—its system prompt—configured via the ADK's instruction parameter2. It is instructed to act as a senior digital media editor. The prompt directs the model to evaluate each clip for visual clarity, subject movement, narrative relevance, and audio quality. Because the raw footage may contain multilingual dialogue, the pipeline benefits from Gemini's native multilingual support, which can comprehend spoken content across 70 different languages18. The agent cross-references this multilingual analysis with the thematic context gathered in Phase I, ensuring the highest engagement scores are awarded to clips that directly highlight the user's stated theme.

### **Enforcing Deterministic Outputs via Pydantic Schemas**

To bridge the gap between generative reasoning and deterministic video rendering, the framework utilizes Gemini's Structured Output functionality, enforcing a strict JSON schema via Pydantic15. The Curation Agent is configured with a response\_schema that forces the model to return an exact data structure15. This guarantees predictable, type-safe results15.  
Table 2 details the required Pydantic schema enforced upon the Gemini model to generate a programmatic Edit Decision List (EDL):

| JSON Key | Type | Description for the LLM | Enforcement Mechanism |
| :---- | :---- | :---- | :---- |
| file\_reference | String | The unique clip\_id matching the local ingestion map. | Strict inclusion required by ADK schema validator. |
| start\_trim | Float | The exact second the engaging segment begins. | Validated as a float greater than or equal to 0.0. |
| end\_trim | Float | The exact second the engaging segment ends. | Validated as a float strictly greater than start\_trim. |
| scene\_rationale | String | A brief justification of why the clip was selected. | Ensures the model applies qualitative reasoning. |
| transition\_intent | String | Suggested visual transition (e.g., 'cut', 'wipe', 'fade'). | Used by the Rendering Agent in Phase V. |

This structured array serves as the primitive Edit Decision List, which is then immediately passed into the autonomous review loop.

## **Phase III: The Autonomous Internal AI Review Loop**

To guarantee that the final video presented to the human reviewer is rigorously evaluated and optimized for virality, the ADK framework implements a LoopAgent6. This workflow primitive acts as an internal quality assurance loop. It takes the draft EDL generated by the Curation Agent and passes it through two highly specialized personas: the Reviewer Agent and the Critic Agent.

### **The Reviewer Agent: Algorithmic Optimization**

The Reviewer Agent is instructed to act as a master of YouTube analytics. Its sole responsibility is to evaluate the proposed EDL against the metrics that drive the YouTube recommendation algorithm.  
The Reviewer Agent analyzes the flow of the clips to ensure maximum algorithmic engagement. It checks pacing (e.g., ensuring cuts occur frequently enough to reset viewer attention) and narrative structure. If the Curation Agent placed a slow, low-energy clip at the beginning of the video, the Reviewer Agent will flag this as a retention drop-off risk. It demands high-energy hooks and ensures that the metadata associated with the chosen clips aligns perfectly with algorithmic search trends.

### **The Critic Agent: Human-Centric Evaluation**

While the Reviewer Agent focuses on data and algorithms, the Critic Agent is designed to view the proposed video strictly from a human perspective. Its primary mandate is to evaluate if the video will capture and hold human attention in real-world browsing scenarios.  
The Critic Agent's specific responsibilities include:

* **The 30-Second Hook:** It rigidly analyzes the exact timestamps selected for the first 30 seconds of the video to determine if they will result in a high Average View Duration (AVD). If the intro fails to deliver immediate value based on the user's stated theme, it is flagged.  
* **Thumbnail CTR Viability:** It analyzes the visual frames selected in the EDL to confirm there are compelling, high-contrast, emotionally engaging moments that could serve as a thumbnail to drive a high Click-Through Rate (CTR).

### **Controlling the Loop with ADK Tools**

The autonomous loop continues to cycle until the quality standards are met. This is managed programmatically via specific ADK tools bound to the Critic Agent.  
If the Critic Agent finds the video lacking in human engagement or algorithmic viability, it utilizes the append\_to\_state tool6. This tool writes critical feedback (e.g., "The first 15 seconds are too slow; remove clip\_04 and replace it with the action shot from clip\_09") back into the ADK's session memory, and the loop restarts. The Curation Agent reads this feedback and generates a revised JSON EDL.  
If, and only if, the Critic Agent determines that the video fully satisfies all viral and engagement criteria, it invokes the exit\_loop tool6. This action breaks the autonomous cycle and advances the state of the pipeline to the human review phase.

## **Phase IV: Assembly and the Human-in-the-Loop Review Mechanism**

In standard video production, generating a fully rendered, high-resolution video file solely for draft review is computationally expensive and introduces severe latency into the pipeline. If the human reviewer requests a minor adjustment, the entire rendering process must be repeated. The architectural solution requires separating the timeline assembly from the final render, presenting a proxy or data-representation of the video for approval.

### **Timeline Generation via OpenTimelineIO (OTIO)**

Once the Critic Agent has approved the edit via the exit\_loop tool6, the Review Orchestrator Agent takes the finalized JSON EDL and converts it into an OpenTimelineIO (.otio) file21. OpenTimelineIO serves as a universal interchange format for editorial cut data21.  
By using the opentimelineio Python library as a custom ADK tool, the agent programmatically constructs a timeline object, appending Track objects for video and audio, and populating them with Clip objects containing the local media paths and TimeRange values calculated from the JSON timestamps12.  
The superiority of OTIO over rendering a proxy video lies in its extreme flexibility and zero-latency generation. The generated .otio file can be served to the user and directly imported into professional Non-Linear Editors (NLEs) like DaVinci Resolve, Adobe Premiere Pro, or Final Cut Pro21.

### **Suspending Execution: The ADK Web UI and Bidirectional Streaming**

Within the ADK application, the Review Orchestrator Agent is responsible for pausing the pipeline to await user approval. The ADK framework excels at managing session state across temporal interruptions. The simplest implementation involves the ADK's built-in web interface (launched via the adk web command)16.  
The agent can send a message through this chat interface: "I have prepared the highly optimized draft timeline. You can review the OpenTimelineIO file at \[path\]. Type 'Approve' to proceed to final rendering, or provide feedback for revisions."27.  
For a significantly more advanced and engaging review process, the ADK framework supports the Gemini Multimodal Live API via bidirectional streaming18. By utilizing the ADK's LiveRequestQueue and run\_live() event loop, the system can establish a persistent WebSocket connection30. This allows the user to speak into their microphone to provide feedback (e.g., "The pacing is too slow, tighten all the cuts by half a second")29.  
**Triggering the Workflow Again:** As per the user's specifications, if the human is not satisfied with the presented edit, they can decline it. The Review Orchestrator Agent will append the human's feedback to the session state and re-trigger the LoopAgent6. The Curation, Reviewer, and Critic agents will then process the human's specific complaints and generate a new draft.  
Once the user explicitly issues an approval command, the pipeline unlocks, advancing the session state and triggering the final rendering phase.

## **Phase V: High-Fidelity Rendering and Transition Enhancement**

Upon user approval, the pipeline transitions from editorial decision-making to heavy computational rendering. The Google ADK itself is an orchestration layer; it does not contain native engines for processing raw video bytes2. For high-engagement videos requiring complex transitions, rapid generation, and strict memory management, the Enhancement and Rendering Agent bypasses Python-level frame manipulation (like MoviePy) entirely33. Instead, the ADK tool acts as a sophisticated wrapper that constructs and executes raw FFmpeg command-line instructions via Python's subprocess module11.

### **Orchestrating Smooth Transitions with FFmpeg xfade**

To address the user's desire for smooth, professional transitions, the Rendering Agent utilizes FFmpeg's highly advanced xfade (crossfade) filter. Introduced in modern FFmpeg builds, xfade provides a single-filter solution for blending two video streams37.  
The xfade filter natively supports over forty distinct transition animations out of the box, which are highly conducive to modern YouTube aesthetics and retention strategies39.  
Table 3 categorizes the primary transition families available via the xfade filter that the ADK Rendering Agent can apply based on the EDL's transition\_intent:

| Transition Family | Available xfade Parameters | Aesthetic Application for YouTube |
| :---- | :---- | :---- |
| **Dissolves** | fade, fadeblack, fadewhite, dissolve, pixelize | Ideal for indicating a passage of time or a shift in narrative tone38. |
| **Directional Wipes** | wipeleft, wiperight, wipeup, wipedown | Useful for fast-paced vlogs or travel montages to maintain kinetic energy23. |
| **Cinematic Slides** | slideleft, slideright, smoothdown, smoothup | Creates a polished, high-production-value feel often used in corporate or tech review videos37. |
| **Geometric Reveals** | circlecrop, rectcrop, circleopen, circleclose | Highly engaging, abrupt transitions suited for comedic timing or dramatic reveals38. |

The engineering complexity in automating xfade via Python lies entirely in calculating the temporal parameters: duration and offset41. The ADK Rendering Agent parses the approved JSON EDL and dynamically calculates the cumulative duration of the timeline to construct the highly specific filter\_complex strings required by FFmpeg.

### **Audio Ducking, Crossfading, and Auto-Captioning**

High-engagement YouTube videos require an audio polish that perfectly matches the visual transitions. The ADK Rendering Agent is programmed to append the acrossfade audio filter in parallel with every xfade video filter to ensure smooth audio blending. Furthermore, if the system is configured to add a continuous background music track beneath the compiled video, the agent implements FFmpeg's sidechaincompress filter43. This technique automatically compresses (lowers the volume of) the background music whenever a person is speaking44.  
To maximize viewer retention, burned-in subtitles are a mandatory enhancement. The final step of the Rendering Agent incorporates OpenAI's Whisper AI into the post-production pipeline45. Whisper automatically detects the spoken language, transcribes the dialogue, and generates a highly accurate, timestamped SubRip Subtitle (.srt) file47. The agent then executes a final FFmpeg pass using the subtitles video filter (e.g., \-vf subtitles=captions.srt) to permanently burn the text into the video frames, complete with engaging styles and colors45.

## **System Security and the ADK Callback Architecture**

Deploying autonomous AI agents that possess the capability to execute heavy local terminal commands—such as invoking FFmpeg binaries with complex string arguments—introduces significant operational and security risks. The Google ADK provides advanced, enterprise-grade mechanisms to secure this pipeline via its robust Callback architecture49.  
Callbacks within the ADK allow developers to hook directly into the agent's execution lifecycle. They function as interceptors, enabling the system to monitor, modify, or halt data at predefined stages of the runtime event loop49. The proposed video architecture utilizes several key callback strategies to guarantee operational safety.  
Table 4 outlines the required ADK Callbacks necessary to secure the automated video pipeline:

| Callback Function | Execution Phase | Architectural Purpose in Video Pipeline | Security Mechanism |
| :---- | :---- | :---- | :---- |
| before\_model\_callback | Prior to the LLM generating a response. | Acts as an initial input guardrail.49. | Prevents user prompt injection attacks that might try to force the agent to ignore the local media folder and download malicious external payloads. |
| before\_tool\_callback | Prior to a Python tool (like FFmpeg execution) running. | Validates arguments generated by the model before they touch the operating system49. | Strictly enforces that the requested FFmpeg command only contains permitted flags and blocks any attempts to read or write to directories outside the designated video staging folder54. |
| after\_tool\_callback | Immediately after a tool completes execution. | Evaluates the tool's output for system failures40. | If FFmpeg returns a compilation error, this callback traps the terminal error and passes a formatted error message back to the LLM for automatic self-correction. |
| after\_model\_callback | After the LLM replies but before the workflow advances. | Validates structured data integrity53. | Ensures that the generated JSON for the Edit Decision List strictly conforms to the expected timecode schema before passing it to the OpenTimelineIO generator38. |

By implementing the before\_tool\_callback, the framework ensures an air-tight boundary between generative reasoning and system execution. This layered defense mechanism is what elevates the ADK from a prototyping tool to a production-ready enterprise framework.

## **Conclusion**

The creation of a highly engaging YouTube video via an automated workflow requires far more than basic API stitching or simplistic prompt chains; it demands a rigorous, multi-layered architectural framework. By leveraging the Google Agent Development Kit (ADK), developers can orchestrate a deterministic, multi-agent pipeline that transforms abstract generative reasoning into tangible, high-fidelity media engineering.  
By anchoring the ingestion phase in secure local directory reading, the system guarantees unthrottled access to source media. Through the integration of an autonomous LoopAgent, the system enforces strict algorithmic and human-centric quality standards, iterating the cut internally before it ever reaches a human supervisor. Crucially, by bridging the AI realm to professional editing environments via OpenTimelineIO, the system honors the necessity of final human review without incurring rendering penalties. Finally, by delegating the ultimate assembly to optimized FFmpeg workflows—complete with mathematically precise xfade transitions, dynamic audio ducking, and Whisper-generated burned-in captions—the framework guarantees an output that meets the high-retention aesthetic standards of modern digital platforms. This seamless convergence of secure agentic orchestration and low-level media processing represents the definitive future of automated, scalable video production.

#### **Works cited**

> 1. Agent Development Kit | Gemini Enterprise Agent Platform, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)  
> 2. Agent Development Kit (ADK) \- Agent Development Kit (ADK), [https://adk.dev/](https://adk.dev/)  
> 3. What is Google ADK (Agent Development Kit) \- GeeksforGeeks, [https://www.geeksforgeeks.org/artificial-intelligence/what-is-google-adk-agent-development-kit/](https://www.geeksforgeeks.org/artificial-intelligence/what-is-google-adk-agent-development-kit/)  
> 4. Google Agent Development Kit (ADK): No-Code vs Code-First Agents, [https://medium.com/@rohitobrai11/google-agent-development-kit-adk-no-code-vs-code-first-agents-28ccc3f44bbb](https://medium.com/@rohitobrai11/google-agent-development-kit-adk-no-code-vs-code-first-agents-28ccc3f44bbb)  
> 5. Building AI Agents with Google ADK (Agent Development Kit), [https://agentswarms.fyi/blog/google-adk-build-ai-agents](https://agentswarms.fyi/blog/google-adk-build-ai-agents)  
> 6. Build Multi-Agent Systems with ADK \- Codelabs, [https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk)  
> 7. GitHub \- Tanvik-hub/google-adk-sequential-multi-agent-workflow, [https://github.com/Tanvik-hub/google-adk-sequential-multi-agent-workflow](https://github.com/Tanvik-hub/google-adk-sequential-multi-agent-workflow)  
> 8. A Developer's Guide to Multi-Agent Systems with ADK \- Google Cloud, [https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk](https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk)  
> 9. Kjdragan/google-adk-tutorial \- GitHub, [https://github.com/Kjdragan/google-adk-tutorial](https://github.com/Kjdragan/google-adk-tutorial)  
> 10. Building AI Agents with ADK: Empowering with Tools \- Codelabs, [https://codelabs.developers.google.com/devsite/codelabs/build-agents-with-adk-empowering-with-tools](https://codelabs.developers.google.com/devsite/codelabs/build-agents-with-adk-empowering-with-tools)  
> 11. Automating Video Editing With Python: MoviePy and FFmpeg, [https://videobycode.com/articles/automating-video-editing-with-python-moviepy-and-ffmpeg-pipelines/](https://videobycode.com/articles/automating-video-editing-with-python-moviepy-and-ffmpeg-pipelines/)  
> 12. otio-discussion@lists.aswf.io | Guidance on generated a fcp xml via, [https://lists.aswf.io/g/otio-discussion/topic/guidance\_on\_generated\_a\_fcp/96805531](https://lists.aswf.io/g/otio-discussion/topic/guidance_on_generated_a_fcp/96805531)  
> 13. Sequential Workflow using Google ADK Gemini, Fast API, Streamlit, [https://dev.to/omerberatsezer/multi-agent-sequential-workflow-using-google-adk-gemini-fast-api-streamlit-create-game-kdk](https://dev.to/omerberatsezer/multi-agent-sequential-workflow-using-google-adk-gemini-fast-api-streamlit-create-game-kdk)  
> 14. Gemini Interactions API \- Google AI for Developers, [https://ai.google.dev/api/interactions-api](https://ai.google.dev/api/interactions-api)  
> 15. Structured outputs \- Interactions API \- Google AI for Developers, [https://ai.google.dev/gemini-api/docs/structured-output](https://ai.google.dev/gemini-api/docs/structured-output)  
> 16. Multi-Agent Example using Google's Agent Development Kit (ADK), [https://medium.com/@imranburki.ib/multi-agent-example-using-googles-agent-development-kit-adk-500312361ebb](https://medium.com/@imranburki.ib/multi-agent-example-using-googles-agent-development-kit-adk-500312361ebb)  
> 17. Python \- Agent Development Kit (ADK), [https://adk.dev/get-started/python/](https://adk.dev/get-started/python/)  
> 18. Gemini Live API overview \- Google AI for Developers, [https://ai.google.dev/gemini-api/docs/live-api](https://ai.google.dev/gemini-api/docs/live-api)  
> 19. Gemini SDK Track Part 2: Text Generation & Structured Outputs, [https://www.wasilzafar.com/pages/series/ai-app-dev-xtreme/ai-app-dev-sdk-gemini-part02-text-structured.html](https://www.wasilzafar.com/pages/series/ai-app-dev-xtreme/ai-app-dev-sdk-gemini-part02-text-structured.html)  
> 20. Google Gemini Tutorial: Structured Outputs with Instructor, [https://python.useinstructor.com/integrations/google/](https://python.useinstructor.com/integrations/google/)  
> 21. Export OpenTimelineIO (OTIO) from Eddie AI, [https://www.heyeddie.ai/exports/otio](https://www.heyeddie.ai/exports/otio)  
> 22. Video tracks order when importing/exporting as OpenTimelineIO in, [https://discuss.kde.org/t/video-tracks-order-when-importing-exporting-as-opentimelineio-in-kdenlive-25/34745](https://discuss.kde.org/t/video-tracks-order-when-importing-exporting-as-opentimelineio-in-kdenlive-25/34745)  
> 23. OpenTimelineIO/otio-fcpx-xml-adapter \- GitHub, [https://github.com/OpenTimelineIO/otio-fcpx-xml-adapter](https://github.com/OpenTimelineIO/otio-fcpx-xml-adapter)  
> 24. opentimelineio.adapters.fcp\_xml \- Read the Docs, [https://opentimelineio.readthedocs.io/en/v0.15/api/python/opentimelineio.adapters.fcp\_xml.html](https://opentimelineio.readthedocs.io/en/v0.15/api/python/opentimelineio.adapters.fcp_xml.html)  
> 25. OpenTimelineIO – What It Is and What It Does | Larry Jordan, [https://larryjordan.com/articles/opentimelineio-now-supported-on-final-cut-premiere-and-resolve/](https://larryjordan.com/articles/opentimelineio-now-supported-on-final-cut-premiere-and-resolve/)  
> 26. Getting Started with Google Agent Development Kit (ADK), [https://geshan.com.np/blog/2026/05/google-adk-tutorial/](https://geshan.com.np/blog/2026/05/google-adk-tutorial/)  
> 27. Build your first AI Agent with ADK \- Agent Development Kit by Google, [https://dev.to/marianocodes/build-your-first-ai-agent-with-adk-agent-development-kit-by-google-409b](https://dev.to/marianocodes/build-your-first-ai-agent-with-adk-agent-development-kit-by-google-409b)  
> 28. How to Use Google ADK for AI Agent Development \- Coursera, [https://www.coursera.org/articles/how-to-use-google-adk](https://www.coursera.org/articles/how-to-use-google-adk)  
> 29. Live and voice agents \- Agent Development Kit (ADK), [https://adk.dev/live/](https://adk.dev/live/)  
> 30. Part 1\. Intro to streaming \- Agent Development Kit (ADK), [https://adk.dev/live/dev-guide/part1/](https://adk.dev/live/dev-guide/part1/)  
> 31. Google ADK \+ Vertex AI Live API \- Medium, [https://medium.com/google-cloud/google-adk-vertex-ai-live-api-125238982d5e](https://medium.com/google-cloud/google-adk-vertex-ai-live-api-125238982d5e)  
> 32. Introduction to ADK Gemini Live API Toolkit \- Codelabs, [https://codelabs.developers.google.com/intro-to-adk-live](https://codelabs.developers.google.com/intro-to-adk-live)  
> 33. Concat videos too slow using Python MoviePY \- Stack Overflow, [https://stackoverflow.com/questions/56413813/concat-videos-too-slow-using-python-moviepy](https://stackoverflow.com/questions/56413813/concat-videos-too-slow-using-python-moviepy)  
> 34. MoviePy: Video Editing as a Python Pipeline \- Florian Narr, [https://www.codeline.co/thoughts/repo-review/2025/moviepy-video-editing-with-python](https://www.codeline.co/thoughts/repo-review/2025/moviepy-video-editing-with-python)  
> 35. Efficiency Disparity Between MoviePy and FFmpeg \#2165 \- GitHub, [https://github.com/Zulko/moviepy/issues/2165](https://github.com/Zulko/moviepy/issues/2165)  
> 36. Python FFmpeg Automation vs. MoviePy for Video Editing, [https://software.reibuys.com/python-ffmpeg-automation-vs-moviepy-for-video-editing/](https://software.reibuys.com/python-ffmpeg-automation-vs-moviepy-for-video-editing/)  
> 37. crossfade between 2 videos using ffmpeg \- Super User, [https://superuser.com/questions/778762/crossfade-between-2-videos-using-ffmpeg](https://superuser.com/questions/778762/crossfade-between-2-videos-using-ffmpeg)  
> 38. FFmpeg xfade Transitions: The Complete Guide (30+ Examples), [http://www.ffmpeglab.com/articles/ffmpeg-xfade-transitions-guide.html](http://www.ffmpeglab.com/articles/ffmpeg-xfade-transitions-guide.html)  
> 39. Crossfade Between Clips with FFmpeg xfade (No Editor), [https://www.ffmpeg-micro.com/blog/crossfade-between-clips-with-ffmpeg-xfade-no-editor](https://www.ffmpeg-micro.com/blog/crossfade-between-clips-with-ffmpeg-xfade-no-editor)  
> 40. 3 Methods you need to know about FFmpeg transition animation, [https://donglumail.medium.com/3-methods-you-need-to-know-for-ffmpeg-transition-animation-7d2ea8f7ced7](https://donglumail.medium.com/3-methods-you-need-to-know-for-ffmpeg-transition-animation-7d2ea8f7ced7)  
> 41. Xfade – FFmpeg, [https://trac.ffmpeg.org/wiki/Xfade](https://trac.ffmpeg.org/wiki/Xfade)  
> 42. XFade Video Transitions Examples, [https://ffmpegbyexample.com/examples/j2ddvy12/xfade\_video\_transitions\_examples/](https://ffmpegbyexample.com/examples/j2ddvy12/xfade_video_transitions_examples/)  
> 43. \[FFmpeg-user\] Audio ducking of input music based on vocals in, [https://ffmpeg.org/pipermail/ffmpeg-user/2018-August/040933.html](https://ffmpeg.org/pipermail/ffmpeg-user/2018-August/040933.html)  
> 44. FFmpeg Filters Documentation, [https://ffmpeg.org/ffmpeg-filters.html](https://ffmpeg.org/ffmpeg-filters.html)  
> 45. Transcribe Any Video with Whisper AI & Python \- YouTube, [https://www.youtube.com/watch?v=9RGlzZ5mr1s](https://www.youtube.com/watch?v=9RGlzZ5mr1s)  
> 46. Generate SRT Subtitles Locally with Whisper: Free & Private, [https://localaimaster.com/blog/local-ai-subtitles-whisper](https://localaimaster.com/blog/local-ai-subtitles-whisper)  
> 47. How to Automate Subtitle Generation with Python and Whisper AI, [https://software.reibuys.com/how-to-automate-subtitle-generation-with-python-and-whisper-ai/](https://software.reibuys.com/how-to-automate-subtitle-generation-with-python-and-whisper-ai/)  
> 48. Run Whisper audio transcriptions with one FFmpeg command, [https://medium.com/@vpalmisano/run-whisper-audio-transcriptions-with-one-ffmpeg-command-c6ecda51901f](https://medium.com/@vpalmisano/run-whisper-audio-transcriptions-with-one-ffmpeg-command-c6ecda51901f)  
> 49. Google ADK Tutorial: Build Your First AI Agent \[2026\] \- Kunal Ganglani, [https://www.kunalganglani.com/blog/google-adk-tutorial-first-agent](https://www.kunalganglani.com/blog/google-adk-tutorial-first-agent)  
> 50. Quick Guide to ADK Callbacks \- Mete Atamel, [https://atamel.dev/posts/2025/11-03\_quick\_guide\_adk\_callbacks/](https://atamel.dev/posts/2025/11-03_quick_guide_adk_callbacks/)  
> 51. Google ADK Masterclass Part 8: Callbacks and Agent Lifecycle, [https://saptak.in/writing/2025/05/10/google-adk-masterclass-part8](https://saptak.in/writing/2025/05/10/google-adk-masterclass-part8)  
> 52. Guardrails via before\_model\_callback — implementation patterns, [https://github.com/google/adk-python/discussions/4963](https://github.com/google/adk-python/discussions/4963)  
> 53. Callbacks | CX Agent Studio \- Google Cloud Documentation, [https://docs.cloud.google.com/gemini-enterprise-cx/cx-agent-studio/callback](https://docs.cloud.google.com/gemini-enterprise-cx/cx-agent-studio/callback)  
> 54. Understanding Callbacks in Google ADK: A Practical Guide \- Medium, [https://medium.com/@merajhussain1/understanding-callbacks-in-google-adk-a-practical-guide-342e41a28480](https://medium.com/@merajhussain1/understanding-callbacks-in-google-adk-a-practical-guide-342e41a28480)