const robotArm = document.getElementById("robot-arm");

const frames = [
    [
        "    .----.            ",
        "    ( HI! )    o       ",
        "     '----'     \\      ",
        "       +------+  \\     ",
        "       | [] []|  /      ",
        "       +------+         ",
        "      /|======|         ",
        "     / |______|         "
    ],
    [   
        "    .----.            ", 
        "    ( HI! )     o      ",
        "     '----'     /      ",
        "        +------+ /       ",
        "        | [] []| /        ",
        "       +------+         ",
        "      /|======|         ",
        "     / |______|         "
    ]
];

const sequence = [0, 1];
const delays = [620, 620];

function renderFrame(index) {
    robotArm.textContent = frames[index].join("\n");
}

function startAnimation() {
    if (!robotArm) {
        return;
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let step = 0;

    renderFrame(sequence[step]);

    if (reducedMotion.matches) {
        return;
    }

    function tick() {
        step = (step + 1) % sequence.length;
        renderFrame(sequence[step]);
        window.setTimeout(tick, delays[step]);
    }

    window.setTimeout(tick, 1000);
}

startAnimation();
