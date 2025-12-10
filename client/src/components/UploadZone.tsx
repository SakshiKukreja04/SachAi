import { useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { Upload, Link as LinkIcon, Loader2 } from 'lucide-react';
import { Button } from './ui/button';

interface UploadZoneProps {
  onUpload: (file: File | null, url: string | null) => void;
  isLoading?: boolean;
}

export const UploadZone = ({ onUpload, isLoading }: UploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [url, setUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
      setUrl('');
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
      setUrl('');
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      onUpload(selectedFile, null);
    } else if (url.trim()) {
      onUpload(null, url.trim());
    }
  };

  const canAnalyze = selectedFile || url.trim();

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Drag & Drop Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={cn(
          'relative rounded-xl border-2 border-dashed p-8 transition-all duration-300',
          'bg-card/40 backdrop-blur-sm',
          isDragging
            ? 'border-primary bg-primary/10 scale-[1.02]'
            : 'border-border hover:border-primary/50 hover:bg-card/60',
          'group cursor-pointer'
        )}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="flex flex-col items-center gap-4 text-center">
          <div className={cn(
            'w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300',
            'bg-gradient-to-br from-primary/20 to-secondary/20',
            isDragging ? 'scale-110 glow-primary' : 'group-hover:scale-105'
          )}>
            <Upload className={cn(
              'w-8 h-8 transition-colors duration-300',
              isDragging ? 'text-primary' : 'text-muted-foreground group-hover:text-primary'
            )} />
          </div>
          
          {selectedFile ? (
            <div className="space-y-1">
              <p className="text-foreground font-medium">{selectedFile.name}</p>
              <p className="text-muted-foreground text-sm">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="text-foreground font-medium">
                Drop your video here or click to browse
              </p>
              <p className="text-muted-foreground text-sm">
                Supports MP4, MOV, AVI up to 500MB
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className="flex items-center gap-4">
        <div className="flex-1 h-px bg-border" />
        <span className="text-muted-foreground text-sm">or</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      {/* URL Input */}
      <div className="relative">
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground">
          <LinkIcon className="w-5 h-5" />
        </div>
        <input
          type="url"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            if (e.target.value) setSelectedFile(null);
          }}
          placeholder="Paste video URL (YouTube, Vimeo, etc.)"
          className={cn(
            'w-full h-12 pl-12 pr-4 rounded-lg',
            'bg-card/60 backdrop-blur-sm border border-border',
            'text-foreground placeholder:text-muted-foreground',
            'focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent',
            'transition-all duration-300'
          )}
        />
      </div>

      {/* Analyze Button */}
      <Button
        variant="hero"
        size="xl"
        className="w-full"
        disabled={!canAnalyze || isLoading}
        onClick={handleAnalyze}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          'Analyze Video'
        )}
      </Button>

      {/* Extension CTA */}
      <Button variant="glass" className="w-full" size="lg">
        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
        Add Browser Extension
      </Button>
    </div>
  );
};
