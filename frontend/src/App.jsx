import ImageFiltering from './components/imagefiltering';
import ThreeDReconstruction from './components/3dconstruction';
import ImageStitching from './components/imagestitching';
import './App.css';

function App() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-6 text-indigo-900">Computer Vision Midterm Project</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ImageFiltering />
        <ThreeDReconstruction />
        <ImageStitching />
      </div>
    </div>
  );
}

export default App;
