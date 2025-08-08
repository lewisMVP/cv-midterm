import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Layers, Grid3x3, Sparkles, ChevronDown, Github, Globe } from 'lucide-react';
import ImageFiltering from './components/imagefiltering';
import ThreeDReconstruction from './components/3dconstruction';
import ImageStitching from './components/imagestitching';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('filtering');
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const tabs = [
    { id: 'filtering', label: 'Image Filtering', icon: Sparkles },
    { id: '3d', label: '3D Reconstruction', icon: Layers },
    { id: 'stitching', label: 'Image Stitching', icon: Grid3x3 },
  ];

  const scrollToContent = () => {
    document.getElementById('content').scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="app-container">
      {/* Navigation Bar */}
      <nav className={`nav-apple px-6 py-4 ${isScrolled ? 'scrolled' : ''}`}>
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              initial={{ rotate: 0 }}
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
              className="w-10 h-10 bg-gradient-to-br from-apple-blue to-apple-purple rounded-full flex items-center justify-center"
            >
              <Camera className="w-5 h-5 text-white" />
            </motion.div>
            <span className="text-xl font-semibold bg-gradient-to-r from-apple-blue to-apple-purple bg-clip-text text-transparent">
              Vision Lab
            </span>
          </div>
          <div className="flex items-center gap-6">
            <a href="https://github.com/lewisMVP/cv-midterm" target="_blank" rel="noopener noreferrer" 
               className="text-apple-gray-600 hover:text-apple-blue transition-colors">
              <Github className="w-5 h-5" />
            </a>
            <a href="#" className="text-apple-gray-600 hover:text-apple-blue transition-colors">
              <Globe className="w-5 h-5" />
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="max-w-4xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <h1 className="hero-title">
              Computer Vision
              <br />
              <span className="gradient-text">Reimagined</span>
            </h1>
            <p className="hero-subtitle">
              Advanced image processing techniques with traditional computer vision methods.
              No deep learning. Pure algorithmic elegance.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex justify-center gap-4"
          >
            <button 
              onClick={scrollToContent}
              className="btn-apple flex items-center gap-2"
            >
              Get Started
              <ChevronDown className="w-4 h-4" />
            </button>
            <a 
              href="https://github.com/lewisMVP/cv-midterm"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-apple-secondary flex items-center gap-2"
            >
              <Github className="w-4 h-4" />
              View Source
            </a>
          </motion.div>
        </div>

        {/* Animated background shapes */}
        <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
          <motion.div
            animate={{
              x: [0, 100, 0],
              y: [0, -100, 0],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute top-20 left-10 w-72 h-72 bg-apple-blue/5 rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              x: [0, -100, 0],
              y: [0, 100, 0],
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute bottom-20 right-10 w-96 h-96 bg-apple-purple/5 rounded-full blur-3xl"
          />
        </div>
      </section>

      {/* Main Content */}
      <section id="content" className="max-w-7xl mx-auto px-6 pb-20">
        {/* Tab Navigation */}
        <div className="flex justify-center mb-12">
          <div className="tab-nav inline-flex">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <span className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    {tab.label}
                  </span>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            {activeTab === 'filtering' && <ImageFiltering />}
            {activeTab === '3d' && <ThreeDReconstruction />}
            {activeTab === 'stitching' && <ImageStitching />}
          </motion.div>
        </AnimatePresence>
      </section>

      {/* Footer */}
      <footer className="border-t border-apple-gray-200 py-12 mt-20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-apple-blue to-apple-purple rounded-full flex items-center justify-center">
                <Camera className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm text-apple-gray-600">
                © 2025 Vision Lab. INS3155 Computer Vision Project.
              </span>
            </div>
            <div className="flex gap-6 text-sm">
              <a href="#" className="text-apple-gray-600 hover:text-apple-blue transition-colors">
                VNU-IS
              </a>
              <a href="https://github.com/lewisMVP/cv-midterm" className="text-apple-gray-600 hover:text-apple-blue transition-colors">
                GitHub
              </a>
              <a href="#" className="text-apple-gray-600 hover:text-apple-blue transition-colors">
                Documentation
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;