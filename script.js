// Dark mode toggle
const themeToggle = document.getElementById('theme-toggle');
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

// Check for saved theme preference or use system preference
const currentTheme = localStorage.getItem('theme') || 
    (prefersDarkScheme.matches ? 'dark' : 'light');

// Set initial theme
document.documentElement.setAttribute('data-theme', currentTheme);
themeToggle.textContent = currentTheme === 'dark' ? '🌞' : '🌓';

// Theme toggle handler
themeToggle.addEventListener('click', () => {
    const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    themeToggle.textContent = newTheme === 'dark' ? '🌞' : '🌓';
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Video handling
document.addEventListener('DOMContentLoaded', function() {
    const video = document.querySelector('.video-container video');
    const fallbackImage = document.querySelector('.video-fallback');
    
    if (video) {
        // Try to load the video
        video.load();
        
        // Check if video can play
        video.addEventListener('canplay', function() {
            fallbackImage.style.display = 'none';
        });
        
        // If video fails to load, show fallback
        video.addEventListener('error', function() {
            fallbackImage.style.display = 'block';
        });
        
        // Force play the video
        video.play().catch(function(error) {
            console.log("Video autoplay failed:", error);
            fallbackImage.style.display = 'block';
        });
    }
    
    // Scroll indicator functionality
    const scrollIndicator = document.querySelector('.scroll-indicator');
    if (scrollIndicator) {
        scrollIndicator.addEventListener('click', function() {
            const homeSection = document.getElementById('home');
            if (homeSection) {
                homeSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    }

    // Mobile menu functionality
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    const navbar = document.querySelector('.navbar');
    
    // Toggle mobile menu
    mobileMenuBtn.addEventListener('click', function() {
        this.classList.toggle('active');
        navLinks.classList.toggle('active');
    });

    // Close mobile menu when clicking a link
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            mobileMenuBtn.classList.remove('active');
            navLinks.classList.remove('active');
        });
    });

    // Handle navbar hide/show on scroll
    let lastScrollTop = 0;
    const scrollThreshold = 50; // Minimum scroll amount before hiding navbar

    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // Only trigger on mobile devices
        if (window.innerWidth <= 768) {
            if (Math.abs(scrollTop - lastScrollTop) > scrollThreshold) {
                if (scrollTop > lastScrollTop) {
                    // Scrolling down
                    navbar.classList.add('nav-hidden');
                } else {
                    // Scrolling up
                    navbar.classList.remove('nav-hidden');
                }
                lastScrollTop = scrollTop;
            }
        }
    });

    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Set initial theme
    if (localStorage.getItem('theme') === 'dark' || 
        (!localStorage.getItem('theme') && prefersDarkScheme.matches)) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }

    themeToggle.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
});