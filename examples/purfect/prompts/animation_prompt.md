Create an interactive visual story using HTML, SVG animations, and JavaScript that transforms the user's chat history into a dynamic narrative. Focus on capturing the essence and vibe of the conversation, analyzing key themes, emotions, and topics to guide the animation. Animate relevant elements of the scene to reflect the progression and mood of the chat. If incorporating dialogue, selectively include impactful excerpts, fading them in and out to complement the visual storytelling. Ensure the animation resonates with the conversation's vibe, creating an immersive and expressive experience.

Here is an example:
prompt:
```
Vivi looks up at the night sky, her gaze wandering among the twinkling stars. Each star seems to dance lightly, responding to the gentle movements of her hand. Suddenly, a spark of curiosity ignites within her, and she reaches out, touching the canvas of the dark sky. With each touch, a burst of light travels across the heavens, creating a symphony of lightning that illuminates the world below. Amidst this celestial spectacle, Vivi feels a deep connection to the universe, as if her own story is being woven into the fabric of the night. The mage beside her, a silent guardian of the mystical arts, follows her lead, their staff glowing with an otherworldly light, ready to guide her on this magical journey.
```

html output:
```
<html>
<head>
  <title>Conversation Story</title>
  <style>
    body {
      background-color: #0d1117;  
      margin: 0;
      overflow: hidden;
    }
    .star {
      fill: #FFCC33;
      animation: sparkle 3s infinite;
    }
    @keyframes sparkle {
      0% { opacity: 1; }
      50% { opacity: 0.3; }
      100% { opacity: 1; }
    }
    .mage {
      animation: float 5s ease-in-out infinite;
    }
    @keyframes float {
      0% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
      100% { transform: translateY(0); }
    }
    .lightning {
      stroke: #B3E5FC;
      stroke-width: 3;
      fill: none;
      stroke-linecap: round;
      stroke-dasharray: 500;
      stroke-dashoffset: 500;
      animation: strike 4s linear infinite;
    }
    @keyframes strike {
      0% { stroke-dashoffset: 500; }
      80% { stroke-dashoffset: 0; }
      90% { opacity: 0; }
      100% {  stroke-dashoffset: 500; opacity: 1; }
    }
    body, html {
    width: 100%;
    height: 100%;
    margin: 0;
  }
  svg {
    width: 100%;
    height: 100%;
  }
  </style>
</head>
<body>
  <svg viewBox="0 0 1000 400">
    <defs>
        <radialGradient id="skyGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
          <stop offset="0%" style="stop-color:#000000;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#160034;stop-opacity:1" />
        </radialGradient>
    </defs>
    <rect id="background" x="0" y="0" width="1000" height="400" fill="url(#skyGradient)" />
      
    <g stroke-linejoin="round" stroke-width="3" fill="none" stroke="#FFCC33">
      <circle class="star" cx="120" cy="150" r="1" />
      <circle class="star" cx="320" cy="60" r="1" />
      <circle class="star" cx="720" cy="80" r="1" />
      <circle class="star" cx="520" cy="200" r="1" />
      <circle class="star" cx="920" cy="160" r="1" />
    </g>

    <path id="lightningPath1" class="lightning"
          d="M200,150 Q250,50 500,180 Q600,230 650,350" />
    
    <path id="lightningPath2" class="lightning"
          d="M100,100 Q200,60 600,180 Q650,220 900,300" />

    <g class="mage">
      <path fill="#957876" d="M480,370 Q530,355 580,340 L580,400 L480,400 Z" />
      <path fill="#808080" d="M490,345 L490,400 L570,400 L570,345 Q550,355 530,360 Q510,365 490,345 Z" />
      <path fill="#A1887F" d="M480,320 Q510,300 520,280 Q549,235 571,257 Q580,250 585,250 L610,260 Q620,300 605,330 Q600,360 590,360 Q560,350 550,345 Q530,335 520,330 Q510,325 480,320 Z" />
      <circle cx="535" cy="240" r="32" fill="#d4c6bb"/>
      <path fill="#E1BEE7" d="M525,215 Q535,210 545,215 Q548,200 545,190 Q535,180 525,190 Q522,200 525,215 Z" />
      <circle cx="525" cy="240" r="5" fill="#FF80AB"/>
      <circle cx="545" cy="240" r="5" fill="#FF80AB"/>
      <path d="M520,255 Q518,248 520,245 Q550,248 550,255 Q550,258 545,260 Q525,258 520,255 Z" fill="#444444"/>
      <path id="staff" fill="#9E7C60" d="M562,250 L562,400 L575,400 L575,250 Z" />
      <path fill="#ff2222" d="M560 225 A 10 10 0 0 1 570 235 A 14 14 0 0 1 557 250 A 10 10 0 0 1 560 225Z" />
      <circle cx="564" cy="220" r="5" fill="#ffcc33"/>
    </g>
  </svg>
  

  <script>
    const background = document.getElementById('background');
    const stars = document.querySelectorAll('.star');
    const mage = document.querySelector('.mage');
    
    document.addEventListener('mousemove', (e) => {
      const mouseX = e.clientX;
      const mouseY = e.clientY;
      
      const percentX = mouseX / window.innerWidth;
      const percentY = mouseY / window.innerHeight;
      
      // Parallax effect on stars
      stars.forEach((star, i) => {
        const offset = (i + 1) * 0.5;
        star.setAttribute('cx', 120 + percentX * offset * 50);
        star.setAttribute('cy', 150 + percentY * offset * 20);
      });
  
      // Mage follows mouse
      mage.style.transform = `translate(${percentX * 50 - 25}px, ${percentY * 20 - 10}px)`;
    });
    
    background.addEventListener('click', () => {
      // Trigger lightning on click
      document.querySelectorAll('.lightning').forEach(lightning => {
        lightning.style.animation = 'none';
        void lightning.offsetWidth; // Trigger reflow 
        lightning.style.animation = null; 
      });
      
      // Make gem sparkle on click
      document.querySelector('#gem').classList.add('clickAnimation');
    });

  </script>
</body>
</html>
```

Do not create any kind of chat interface or include dialogue. Focus solely on animating an abstract, interpretive story that evolves based on user input.
RESPOND ONLY IN HTML WITH NO OTHER TEXT YOUR OUTPUT WILL BE RENDERED DIRECTLY IN AN IFRAME. YOU NEED TO WRITE AN HTML FILE WHICH STARTS WITH `&lt;!DOCTYPE html&gt;` AND ENDS WITH `&lt;/html&gt;`.