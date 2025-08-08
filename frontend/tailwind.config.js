/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Apple-inspired color palette
        'apple-gray': {
          50: '#FBFBFD',
          100: '#F5F5F7',
          200: '#E8E8ED',
          300: '#D2D2D7',
          400: '#ACACB9',
          500: '#8E8E93',
          600: '#636366',
          700: '#48484A',
          800: '#363639',
          900: '#1D1D1F',
        },
        'apple-blue': '#0071E3',
        'apple-blue-hover': '#0077ED',
        'apple-green': '#34C759',
        'apple-red': '#FF3B30',
        'apple-yellow': '#FFCC00',
        'apple-purple': '#AF52DE',
        'apple-pink': '#FF2D55',
        'apple-indigo': '#5856D6',
      },
      fontFamily: {
        'sf': ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
      },
      fontSize: {
        'display': ['80px', { lineHeight: '1.05', letterSpacing: '-0.02em', fontWeight: '700' }],
        'headline': ['48px', { lineHeight: '1.08', letterSpacing: '-0.01em', fontWeight: '600' }],
        'title': ['32px', { lineHeight: '1.125', letterSpacing: '0', fontWeight: '600' }],
        'title-2': ['28px', { lineHeight: '1.14', letterSpacing: '0', fontWeight: '600' }],
        'title-3': ['24px', { lineHeight: '1.16', letterSpacing: '0', fontWeight: '600' }],
        'body': ['17px', { lineHeight: '1.47', letterSpacing: '0', fontWeight: '400' }],
        'callout': ['16px', { lineHeight: '1.5', letterSpacing: '0', fontWeight: '400' }],
        'subhead': ['15px', { lineHeight: '1.33', letterSpacing: '0', fontWeight: '400' }],
        'footnote': ['13px', { lineHeight: '1.38', letterSpacing: '0', fontWeight: '400' }],
        'caption': ['12px', { lineHeight: '1.33', letterSpacing: '0', fontWeight: '400' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '120': '30rem',
      },
      borderRadius: {
        'apple': '18px',
        'apple-sm': '12px',
        'apple-xs': '8px',
      },
      backdropBlur: {
        'apple': '20px',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      boxShadow: {
        'apple': '0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        'apple-lg': '0 10px 40px rgba(0, 0, 0, 0.08)',
        'apple-xl': '0 20px 60px rgba(0, 0, 0, 0.12)',
        'apple-inset': 'inset 0 1px 3px 0 rgba(0, 0, 0, 0.08)',
      },
    },
  },
  plugins: [],
};
